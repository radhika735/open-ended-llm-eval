import numpy as np
import os
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, cohen_kappa_score

def get_aligned_score_arrays(metric = "faithfulness"):
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    summaries_subdirs = ["summaries__gpt-5", "summaries__claude-sonnet-4"]
    flash_subdir = "judge__gemini-2-5-flash"
    pro_subdir = "judge__gemini-2-5-pro"
    pro_scores = []
    flash_scores = []
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for summaries_subdir in summaries_subdirs:
                base_dir = f"live_evaluations/{qu_type}_{filter_stage}_evals"
                
                
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
                            pro_scores.append(pro_score)
                            flash_scores.append(flash_score)
                        else:
                            print(f"Flash not evaluation {qu_type}, {filter_stage}, {summaries_subdir}, {filename}, query: {query}")

    return pro_scores, flash_scores


def calc_similarity(metric = "faithfulness"):
    pro_scores, flash_scores = get_aligned_score_arrays(metric=metric)
    for p,f in zip(pro_scores, flash_scores):
        print(f"Pro: {p}, Flash: {f}")
    print(f"Total aligned scores: {len(pro_scores)}")
    pro_scores = np.array(pro_scores)
    flash_scores = np.array(flash_scores)

    model_a = pro_scores
    model_b = flash_scores
    
    # pearson_corr, _ = pearsonr(model_a, model_b)
    # print(f"Pearson corr: {pearson_corr}")

    # # 2. Spearman rank correlation
    # spearman_corr, _ = spearmanr(model_a, model_b)
    # print(f"Spearman rank corr: {spearman_corr}")

    # 3. Mean Absolute Difference (MAD)
    mad = mean_absolute_error(model_a, model_b)
    print(f"Mean absolute diff: {mad}")

    # 4. Root Mean Squared Error (RMSE)
    rmse = root_mean_squared_error(model_a, model_b)
    print(f"RMSE: {rmse}")

    mean_difference = np.mean(model_b - model_a)
    print(f"Mean difference (model_b - model_a): {mean_difference}")



def icc2_1(scores_matrix):
    """
    Compute ICC(2,1): two-way random, absolute agreement, single rater.
    scores_matrix: numpy array (n_targets x n_raters), e.g. rows=items, cols=models
    """
    n_targets, n_raters = scores_matrix.shape
    
    # Means
    mean_per_target = scores_matrix.mean(axis=1)
    mean_per_rater = scores_matrix.mean(axis=0)
    grand_mean = scores_matrix.mean()
    
    # Sum of squares
    ss_total = ((scores_matrix - grand_mean)**2).sum()
    ss_between_targets = (n_raters * ((mean_per_target - grand_mean)**2)).sum()
    ss_between_raters = (n_targets * ((mean_per_rater - grand_mean)**2)).sum()
    ss_residual = ss_total - ss_between_targets - ss_between_raters
    
    # Degrees of freedom
    df_between_targets = n_targets - 1
    df_between_raters = n_raters - 1
    df_residual = df_between_targets * df_between_raters
    
    # Mean squares
    ms_between_targets = ss_between_targets / df_between_targets
    ms_between_raters = ss_between_raters / df_between_raters
    ms_residual = ss_residual / df_residual
    
    # ICC(2,1) formula
    icc_value = (ms_between_targets - ms_residual) / (
        ms_between_targets +
        (n_raters - 1) * ms_residual +
        (n_raters * (ms_between_raters - ms_residual) / n_targets)
    )
    return icc_value


if __name__ == "__main__":
    calc_similarity()

