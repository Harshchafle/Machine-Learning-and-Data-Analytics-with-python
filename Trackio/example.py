# import trackio as wandb

import trackio
import random 
import time

runs = 3
epochs = 5

def simulate_multiple_runs():
    for run in range(runs):
        trackio.init(project="fake-training", config={
            "epochs": epochs,
            "learning_rate": 0.01,
            "batch_size": 64
        })
        for epoch in range(epochs):
            train_loss = random.uniform(0.2, 1.0)
            train_accuracy = random.uniform(0.6, 0.95)
            val_loss = train_loss - random.uniform(0.01, 0.1)
            val_accuracy = train_accuracy + random.uniform(0.01, 0.05)
            trackio.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
            time.sleep(0.2)
        trackio.finish()
    print("Simulation complete.")
        
simulate_multiple_runs()
trackio.show()  # Display the logged runs in the Trackio UI

## output
