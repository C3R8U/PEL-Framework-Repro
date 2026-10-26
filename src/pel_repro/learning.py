"""Four-component learner profiling and pre-requisite-respecting path planning."""
import heapq
import numpy as np


def weighted_affect(text_score=None, audio_score=None, text_weight=.55, audio_weight=.45):
    parts = [(text_score,text_weight),(audio_score,audio_weight)]
    present = [(float(v), w) for v,w in parts if v is not None and not np.isnan(float(v))]
    if not present: raise ValueError('No observed affect modality')
    if any(not 0 <= v <= 1 for v,_ in present): raise ValueError('Scores must be between 0 and 1')
    return sum(v*w for v,w in present)/sum(w for _,w in present)


def update_profile(old_vector, observed_vector, eta):
    a,b = np.asarray(old_vector,dtype=float), np.asarray(observed_vector,dtype=float)
    if a.shape != b.shape or not 0 <= eta <= 1:
        raise ValueError('Incompatible profile dimensions or eta out of range')
    return ((1-eta)*a+eta*b).tolist()


def task_objective(task, weights):
    """All scores (mastery_gain,time_cost,satisfaction,cognitive_load) scaled 0..1.

    Lower = better; weighted transformation of (maximize gain, minimize time,
    maximize satisfaction, minimize load). No negative graph edge weights.
    """
    keys=('mastery_gain','time_cost','satisfaction','cognitive_load')
    for k in keys:
        if not 0 <= float(task[k]) <= 1: raise ValueError(f'{k} outside [0,1]')
    if abs(sum(weights.values())-1)>1e-8: raise ValueError('Weights must sum to 1')
    return (weights['mastery']*(1-task['mastery_gain']) +
            weights['time']*task['time_cost'] +
            weights['satisfaction']*(1-task['satisfaction']) +
            weights['cognitive_load']*task['cognitive_load'])


def optimize_path(tasks, weights, initially_mastered=()):
    """Find minimum-cost prerequisite-respecting full learning route.

    Exact Dijkstra on subset states, <=16 tasks (exponential), reproducible small
    pedagogical knowledge graphs; not a scalability claim for campus-scale data.
    """
    if len(tasks) > 16: raise ValueError('Exact subset planning limited to 16 tasks')
    by_id={task['id']:task for task in tasks}
    if len(by_id)!=len(tasks): raise ValueError('Duplicate task id')
    for task in tasks:
        if any(p not in by_id and p not in initially_mastered for p in task.get('prerequisites', [])):
            raise ValueError('Unknown prerequisite')
    ids=list(by_id)
    done=set(initially_mastered)
    frontier=[(0.0, frozenset(), ())]
    best={frozenset():0.0}
    while frontier:
        cost,selected,path=heapq.heappop(frontier)
        if cost > best.get(selected,float('inf'))+1e-12: continue
        if len(selected)==len(ids): return {'ordered_task_ids':list(path),'total_cost':cost}
        for k in ids:
            if k in selected: continue
            task=by_id[k]
            if not set(task.get('prerequisites', [])).issubset(done|selected): continue
            future=selected|{k}
            score=cost+task_objective(task, weights)
            if score < best.get(future,float('inf')):
                best[future]=score
                heapq.heappush(frontier,(score,future,path+(k,)))
    raise ValueError('Infeasible knowledge prerequisites (cycle?)')
