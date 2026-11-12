"""Knowledge estimation for Moodle learners.

Turns a course's own graded answers into learner-by-KC knowledge states, in
three phases:

    1  cold start    per-KC Bayesian Knowledge Tracing with fixed priors
    2  training      a deep KT model is trained once enough answers exist
    3  personalised  the deep model is used, but only after it has beaten the
                     Phase 1 estimator on held-out learners of the same course

No benchmark dataset is read here. The benchmark experiment in
``knowledge_tracing/`` justifies the choice of model; it never supplies
weights, KC identities or learner states to this package.
"""
