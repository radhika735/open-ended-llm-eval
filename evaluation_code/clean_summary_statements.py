import os
import json

def basic_folder_rename():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    base_dir = "live_summaries"

    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                orig = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, "stmts")
                new = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, "stmts")
                if os.path.exists(orig):
                    os.rename(orig, new)


def del_eval_by_models_field():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    base_dir = "live_summaries"
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                dir_type = "stmt_gen_annotated"
                full_dir = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, dir_type)
                if os.path.exists(full_dir):
                    for filename in os.listdir(full_dir):
                        filepath = os.path.join(full_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for entry in data:
                            if "eval_by_models" in entry:
                                del entry["eval_by_models"]
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)


def add_two_stage_summary_stmt_gen_field():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    base_dir = "live_summaries"
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                dir_type = "stmt_gen_annotated"
                full_dir = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, dir_type)
                if os.path.exists(full_dir):
                    for filename in os.listdir(full_dir):
                        filepath = os.path.join(full_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for entry in data:
                            if entry.get("gen_summary_stmts_request_sent", None) is True:
                                del entry["gen_summary_stmts_request_sent"]
                                del entry["gen_summary_stmts_received"]
                                entry["gen_summary_stmts_request_made"] = True
                                entry["gen_summary_stmts_received"] = True
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                                





def annotate_summary_statements():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"

    for qu_type in qu_types:
        for filter_stage in filter_stages:
            outer_summaries_dir = os.path.join("live_summaries", f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, "eval_annotated")
            for answering_model_provider in os.listdir(outer_summaries_dir):
                full_summaries_dir = os.path.join(outer_summaries_dir, answering_model_provider)
                for summary_filename in os.listdir(full_summaries_dir):
                    with open(os.path.join(full_summaries_dir, summary_filename),'r', encoding='utf-8') as f:
                        summaries_data = json.load(f)
                    eval_filename = summary_filename.replace("summaries", "eval")
                    eval_filepath = os.path.join("live_evaluations_realtime_api", f"{qu_type}_{filter_stage}_evals", "judge__gemini-2-5-pro", f"summaries_{answering_model_provider}", eval_filename)
                    out_filename = summary_filename.replace("summaries", "statements")
                    outpath = os.path.join("live_summaries", f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, "stmts", answering_model_provider, out_filename)
                    if not os.path.exists(eval_filepath):
                        eval_data = []
                    else:
                        with open(eval_filepath, 'r', encoding='utf-8') as f:
                            eval_data = json.load(f)
                    
                    out_data = []

                    for summary_details in summaries_data:
                        query = summary_details['query']
                        relevant_summary = summary_details["relevant_summary"]
                        statements_found = False
                        summary_statements = None
                        for eval_entry in eval_data:
                            if eval_entry["question_details"]["query"] == query and eval_entry["summary_details"]["relevant_summary"] == relevant_summary:
                                eval_entry["summary_statement"] = relevant_summary
                                # summary statements and cited statements exist for this summary
                                summary_details["summary_statements_generated"] = True
                                statements_found = True
                                summary_statements = eval_entry["evaluation_details"]["summary_statements"]
                                break
                        assembled = {
                            "question_details":{
                                "query": query,
                                "all_relevant_qu_ids": summary_details["all_relevant_action_ids"],
                                "regenerated_qu_ids": summary_details["regenerated_ids"]
                            },
                            "summary_details":{
                                "summary_model": summary_details["model"],
                                "summary_provider": summary_details["provider"],
                                "relevant_summary": summary_details["relevant_summary"],
                                "summary_action_ids": summary_details["summary_action_ids"],
                            }
                        }
                        if statements_found:
                            assembled["summary_details"].update({
                                "summary_statements" : summary_statements,
                            })
                        out_data.append(assembled)
                    
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                    with open(outpath, 'w', encoding='utf-8') as f:
                        json.dump(out_data, f, indent=2, ensure_ascii=False)
                    with open(os.path.join(full_summaries_dir, summary_filename), 'w', encoding='utf-8') as f:
                        json.dump(summaries_data, f, indent=2, ensure_ascii=False)


def swap_dir_hierarchy():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    dir_types = [
        "summaries_tool_calls",
        "stmts_and_cited_stmts",
        "eval_annotated"
    ]
    base_dir = "live_summaries"

    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                for dir_type in dir_types:
                    orig = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, dir_type, answering_mp)
                    new = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, dir_type)
                    if os.path.exists(orig):
                        os.makedirs(os.path.dirname(new), exist_ok=True)
                        os.rename(orig, new)


def reset_summary_statements_flags():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    base_dir = "live_summaries"
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                dir_type = "stmt_gen_annotated"
                full_dir = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, dir_type)
                if os.path.exists(full_dir):
                    for filename in os.listdir(full_dir):
                        filepath = os.path.join(full_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for entry in data:
                            if entry.get("gen_summary_stmts_request_made", None) is True:
                                if entry.get("gen_summary_stmts_received", False) is False:
                                    entry["gen_summary_stmts_request_made"] = False
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)


def add_summary_stmts_model_field():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    base_dir = "live_summaries"
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                dir_type = "stmts"
                full_dir = os.path.join(base_dir, f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, dir_type)
                if os.path.exists(full_dir):
                    for filename in os.listdir(full_dir):
                        filepath = os.path.join(full_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for entry in data:
                            if entry["summary_details"]["relevant_summary"] is not None:
                                if "summary_statements" in entry["summary_details"]:
                                    if len(entry["summary_details"]["summary_statements"]) == 0:
                                        print("\n\n\n")
                                        print(entry["summary_details"]["relevant_summary"])
                                        print(entry["summary_details"]["summary_statements"])

                                # if entry["summary_details"].get("summary_statements_model", None) is None:
                                    # statements = entry["summary_details"]["summary_statements"]
                                    # del entry["summary_details"]["summary_statements"]
                                    # entry["summary_details"]["summary_statements_model"] = "gemini-2.5-pro"
                                    # entry["summary_details"]["summary_statements"] = statements

                                # print(entry["summary_details"]["summary_statements_model"])
                                # print(entry["summary_details"]["summary_statements"][0])
                                # print("\n\n")

                        #     if entry["summary_details"].get("summary_statements_model", None) is None:
                        #         statements = entry["summary_details"]["summary_statements"]
                        #         entry["summary_details"]["summary_statements_model"] = "gemini-2.5-pro"
                        # with open(filepath, 'w', encoding='utf-8') as f:
                        #     json.dump(data, f, indent=2, ensure_ascii=False)



def print_failed_summary_stmts_flag():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_type = "hybrid_cross-encoder"
    answering_mps = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for answering_mp in answering_mps:
                summaries_dir = os.path.join("live_summaries", f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, "stmt_gen_annotated")
                statements_dir = os.path.join("live_summaries", f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, answering_mp, "stmts")
                for statements_filename in sorted(os.listdir(statements_dir)):
                    statements_filepath = os.path.join(statements_dir, statements_filename)
                    with open(statements_filepath, 'r', encoding='utf-8') as f:
                        statements_data = json.load(f)
                    for stmt_entry in statements_data:
                        if stmt_entry["summary_details"]["relevant_summary"] is not None:
                            if "summary_statements" in stmt_entry["summary_details"]:
                                if len(stmt_entry["summary_details"]["summary_statements"]) == 0:
                                    print("\n\n\n")
                                    print(f"File: {statements_filepath}")
                                    print(stmt_entry["summary_details"]["relevant_summary"])
                                    print(stmt_entry["summary_details"]["summary_statements"])

                                    # Find corresponding entry in summaries file
                                    summaries_filepath = os.path.join(summaries_dir, statements_filename.replace("statements", "summaries"))
                                    with open(summaries_filepath, 'r', encoding='utf-8') as f:
                                        summaries_data = json.load(f)
                                    for summary_entry in summaries_data:
                                        if summary_entry["relevant_summary"] == stmt_entry["summary_details"]["relevant_summary"] and summary_entry["query"] == stmt_entry["question_details"]["query"]:
                                            print("Found matching summary entry.")
                                            print(f"gen_summary_stmts_request_made: {summary_entry.get('gen_summary_stmts_request_made', None)}")
                                            print(f"gen_summary_stmts_received: {summary_entry.get('gen_summary_stmts_received', None)}")
                                            break


if __name__ == "__main__":
    ## FUNCTIONS GET OUTDATED QUICKLY - CHECK CAREFULLY BEFORE RUNNING

    print_failed_summary_stmts_flag()
    