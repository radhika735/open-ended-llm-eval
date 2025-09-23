import numpy as np
import os
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, cohen_kappa_score

def get_aligned_score_arrays(metric = "faithfulness"):
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    pro_scores = []
    flash_scores = []
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            base_dir = f"live_evaluations/{qu_type}_{filter_stage}_evals"
            flash_subdir = "judge__gemini-2-5-flash"
            pro_subdir = "judge__gemini-2-5-pro"
            summaries_subdir = "summaries_gpt-5"
            pro_path = os.path.join(base_dir, pro_subdir, summaries_subdir)
            flash_path = os.path.join(base_dir, flash_subdir, summaries_subdir)

            if not os.path.exists(pro_path):
                continue
            
            for filename in sorted(os.listdir(pro_path)):
                with open(os.path.join(pro_path, filename), 'r', encoding='utf-8') as pro_eval_file:
                    pro_evals = json.load(pro_eval_file)
                
                if not os.path.exists(os.path.join(flash_path, filename)):
                    continue
                with open(os.path.join(flash_path, filename), 'r', encoding='utf-8') as flash_eval_file:
                    flash_evals = json.load(flash_eval_file)
                
                for pro_eval in pro_evals:
                    query = pro_eval["question_details"]["query"]
                    for pro_metric_eval in pro_eval["evaluation_details"]["evaluations"]:
                        if pro_metric_eval["metric"] == metric:
                            pro_score = pro_metric_eval["evaluation"]["score"]
                            break

                    flash_score = None
                    for flash_eval in flash_evals:
                        if flash_eval["question_details"]["query"] == query:
                            for flash_metric_eval in flash_eval["evaluation_details"]["evaluations"]:
                                if flash_metric_eval["metric"] == metric:
                                    flash_score = flash_metric_eval["evaluation"]["score"]
                                    break
                            break
                            
                    if flash_score is not None:
                        print(query, pro_score, flash_score)
                        pro_scores.append(pro_score)
                        flash_scores.append(flash_score)

    return pro_scores, flash_scores


def calc_similarity(metric = "faithfulness"):
    pro_scores, flash_scores = get_aligned_score_arrays(metric=metric)
    print(f"pro_scores: {pro_scores}")
    print(f"flash_scores: {flash_scores}")
    pro_scores = np.array(pro_scores)
    flash_scores = np.array(flash_scores)

    model_a = pro_scores
    model_b = flash_scores
    
    pearson_corr, _ = pearsonr(model_a, model_b)
    print(f"Pearson corr: {pearson_corr}")

    # 2. Spearman rank correlation
    spearman_corr, _ = spearmanr(model_a, model_b)
    print(f"Spearman rank corr: {spearman_corr}")

    # 3. Mean Absolute Difference (MAD)
    mad = mean_absolute_error(model_a, model_b)
    print(f"Mean absolute diff: {mad}")

    # 4. Root Mean Squared Error (RMSE)
    rmse = root_mean_squared_error(model_a, model_b)
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    calc_similarity()

