"""
mc_io.py  --  AALpy MarkovChain <-> plain dict (picklable).

AALpy's MarkovChain doesn't implement to_state_setup(), so it can't be
pickled directly. We store a minimal dict the scorer can walk:

    {
      "initial": state_id,
      "states": { state_id: {"output": symbol,
                             "trans": [(target_id, prob), ...]} }
    }
"""
import pickle


def mc_to_dict(model):
    states = {}
    for s in model.states:
        states[s.state_id] = {
            "output": s.output,
            "trans": [(t.state_id, float(p)) for t, p in s.transitions],
        }
    return {"initial": model.initial_state.state_id, "states": states}


def save_mc(model, path):
    with open(path, "wb") as f:
        pickle.dump(mc_to_dict(model), f)


def load_mc(path):
    with open(path, "rb") as f:
        return pickle.load(f)
