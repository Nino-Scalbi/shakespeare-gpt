import argparse
import logging
import yaml
import os
import torch
from datetime import datetime
from src.model import GPT
from src.dataset import get_batch, split_data, decode
from src.utils import get_save_steps, get_paths, MetricsLogger, estimate_loss


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
    return parser.parse_args()



def main(run_id, config_path):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Load config
    logger.info("Loading Config")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    max_iters = config["training"]["max_iters"]
    eval_interval = config["training"]["eval_interval"]
    log_interval = config["training"]["log_interval"]
    save_interval = config["training"]["save_interval"]
    learning_rate = config["training"]["learning_rate"]
    eval_iters = config["training"]["eval_iters"]




    paths = get_paths(cwd = os.getcwd(), run_id=run_id)
    if paths["run"].exists():
        raise FileExistsError(f"Run '{run_id}' already exists at {paths['run']}")


    # import the data
    logger.info("Import Data")
    data = split_data(train_ratio=config["training"]["train_ratio"])

    logger.info("Create Model")
    model = GPT(config['model'])
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


    # Training loop
    logger.info(f"Start Training Loop ({max_iters} iterations)")

    save_steps = get_save_steps(max_steps=max_iters, n_steps=config["training"]["n_save_steps"])


    metrics_logger = MetricsLogger(path=paths["metrics"])

    if 0 in save_steps:
        logger.info("Save initial, untrained weights")
        m.save_weights(step=0, output_dir=paths["weights"])

    for iter in range(1, max_iters + 1):

        #sample a batch of the data
        xb, yb = get_batch(split='train', 
                        data = data,
                        config = config["model"])
        xb, yb = xb.to(device), yb.to(device)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient norm (only when needed)
        should_log = iter % log_interval == 0 or iter in save_steps

        if should_log:
            grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in m.parameters() if p.grad is not None]
            ).item()
        
        
        optimizer.step()

        # Save weights
        if iter in save_steps:
            logger.info(f"Save weights at step {iter}")
            m.save_weights(iter, paths["weights"])


        # Log metrics
        if should_log:
            # TODO: be careful when logging loss since it depends on get_batch. For our analysis we should use a random generator based on a fixed seed.
            losses = estimate_loss(
                model=m,
                data=data, 
                config=config["model"],
                device=device,
                eval_iters=eval_iters
                )
            
            metrics_logger.log(
                step=iter,
                train_loss=losses["train"],
                val_loss=losses["val"],
                grad_norm=grad_norm,
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
    main(run_id=run_id, config_path=args.config)