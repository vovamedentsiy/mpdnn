import os

# def modify_update(update,max_elem_len=3):
#     if "restore_dir" in update:
#         update = update.split("-")[0]
#     elif "path" in update and len(update) > 70:
#         update = update.split("-")[0]
#         update = update.replace("'", "")
#         update = ""
#     else:
#         elems = update.split("_")
#         update = "_".join([e[:max_elem_len] for e in elems[:-1]] + [elems[-1].split("=")[0][:max_elem_len] + elems[-1].split("=")[-1]])
#
#     return update
import traceback


def get_experiment_name(ex_path, _run):
    start_time = _run.start_time.isoformat().replace(':', '.').replace('-', '.')
    exp_name = start_time
    exp_name = exp_name + "-" + ex_path + "-"
    for update in sorted(_run.meta_info["options"]["UPDATE"]):
        # update=modify_update(update)
        if "pretrain_path" in update:
            exp_name += "pretrain_path=" + update.split("=")[1].split("-")[0] + "."
        else:
            exp_name = exp_name + str(update) + "."
    exp_name = exp_name[:-1]

    return exp_name

def get_experiment_dir(ex_path, _run):
    from artemis.fileman.local_dir import get_local_dir
    exp_name = get_experiment_name(ex_path, _run)
    experiment_dir = get_local_dir(os.path.join("experiments",exp_name),True)
    return experiment_dir


def write_error_trace(log_dir, print_too = True, filename="errortrace.txt"):
    file_path = os.path.join(log_dir, filename)
    # assert not os.path.exists(file_path), 'Error trace has already been created in this experiment... Something fishy going on here.'
    with open(file_path, 'a+') as f:
        error_text = traceback.format_exc()
        f.write(error_text)
    if print_too:
        print(error_text)
