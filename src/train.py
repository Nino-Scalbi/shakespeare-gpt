import argparse
import logging
import os
from datetime import datetime

import torch
import yaml

from src.dataset import (DATA_PATH,
                         get_batch,
                         prepare_data
                         )
from src.model import GPT
from src.utils import (MetricsLogger,
                       evaluate,
                       get_paths,
                       get_save_steps,
                       save_run_info,
                       set_seed
                       )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Shakespeare GPT")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run ID. Defaults to a timestamp (YYYYMMDD-HHMM)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                        help="Path to the config file") 
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for reproducibility")
    return parser.parse_args()



def main(run_id, config_path, seed):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Load config
    logger.info("Loading Config")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    max_iters = config["training"]["max_iters"]
    log_interval = config["training"]["log_interval"]
    save_interval = config["training"]["save_interval"]
    learning_rate = config["training"]["learning_rate"]
    block_size = config["model"]["block_size"]
    batch_size = config["model"]["batch_size"]




    paths = get_paths(cwd = os.getcwd(), run_id=run_id)
    if paths["run"].exists():
        raise FileExistsError(f"Run '{run_id}' already exists at {paths['run']}")

    # Set random seed for reproducibility
    set_seed(seed)
    # Separate generator for training batches
    training_gen = torch.Generator().manual_seed(seed)

    # import the data
    logger.info("Import Data")
    train_data, val_data, train_starts, val_starts, vocab_size, decode = prepare_data(config)
    config["model"]["vocab_size"] = vocab_size  # so the saved config.yaml is enough to rebuild the model

    # Resolved config and run metadata
    save_run_info(paths["run"], config, seed, device, DATA_PATH)

    def eval_metrics():
        """Eval-mode losses of the current weights on the fixed windows."""
        return dict(
            train_eval_loss=evaluate(m, train_data, train_starts, block_size, batch_size, device),
            val_loss=evaluate(m, val_data, val_starts, block_size, batch_size, device),
        )

    logger.info("Create Model")
    model = GPT(config['model'])
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(params=m.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=config["training"]["weight_decay"]
                                  )


    # Training loop
    logger.info(f"Start Training Loop ({max_iters} iterations)")

    save_steps = get_save_steps(max_steps=max_iters, n_steps=config["training"]["n_save_steps"])


    metrics_logger = MetricsLogger(path=paths["metrics"])

    if 0 in save_steps:
        # Save initial, untrained model state and evaluate metrics
        logger.info("Save initial, untrained weights")
        m.save_weights(step=0, output_dir=paths["weights"])
        # batch_loss and grad_norm are undefined at step 0: no training batch has been seen yet
        metrics_logger.log(
            step=0,
            batch_loss=None,
            grad_norm=None,
            **eval_metrics(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    for iter in range(1, max_iters + 1):

        #sample a batch of the data
        xb, yb = get_batch(split='train', 
                        data = (train_data, val_data),
                        config = config["model"],
                        generator = training_gen)
        xb, yb = xb.to(device), yb.to(device)

        # evaluate the loss
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # compute batch_loss and grad_norm
        should_log = iter % log_interval == 0 or iter in save_steps
        if should_log:
            batch_loss = loss.item()
            grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in m.parameters() if p.grad is not None]
            ).item()

        optimizer.step()

        # Save weights
        if iter in save_steps:
            logger.info(f"Save weights at step {iter}")
            m.save_weights(iter, paths["weights"])

        # Log metrics. The eval-mode losses are measured on the weights AFTER the update (step iter).
        if should_log:
            metrics_logger.log(
                step=iter,
                batch_loss=batch_loss,
                grad_norm=grad_norm,
                **eval_metrics(),
                learning_rate=optimizer.param_groups[0]["lr"],
            )

        # Persist metrics
        if iter % save_interval == 0:
            metrics_logger.save()

    # Save remaining metrics
    metrics_logger.save()
    logger.info("Training Complete")


    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



if __name__ == "__main__":
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M")
    main(run_id=run_id, config_path=args.config, seed=args.seed)
