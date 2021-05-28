import json
import os
import pandas as pd

from absl import app, flags

from chemberta.utils.molnet_dataloader import get_dataset_info


flags.DEFINE_string(name="run_dir", default=None, help="")

flags.mark_flag_as_required("run_dir")

FLAGS = flags.FLAGS


def main(argv):
    
    for split in ("valid", "test"):
        print(f"\nAggregating metrics for {split}...")
    
        df_list = []

        for dataset in sorted(os.listdir(FLAGS.run_dir)):
            try:
                get_dataset_info(dataset)
                print(dataset)
            except:
                continue

            metrics_file_valid = os.path.join(FLAGS.run_dir, dataset, "results", split, "metrics.json")
            if not os.path.exists(metrics_file_valid):
                print(f"Could not find {metrics_file_valid}")
                continue

            with open(metrics_file_valid, "r") as f:
                metrics_by_run = json.load(f)
                for run, metrics in metrics_by_run.items():
                    if "pearsonr" in metrics:
                        metrics["pearsonr"] = metrics["pearsonr"][0]

            df = pd.DataFrame(metrics_by_run.values())
            df.insert(0, "dataset", dataset)

            df_list.append(df)

        df_all = pd.concat(df_list, axis=0)
        df_mean = df_all.groupby("dataset").mean()
        df_std = df_all.groupby("dataset").std()

        df_mean.to_csv(os.path.join(FLAGS.run_dir, f"metrics_{split}_mean.csv"))
        df_std.to_csv(os.path.join(FLAGS.run_dir, f"metrics_{split}_std.csv"))
        

if __name__ == "__main__":
    app.run(main)
