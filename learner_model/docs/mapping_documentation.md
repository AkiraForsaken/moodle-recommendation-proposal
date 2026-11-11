# KC taxonomy and Moodle mapping status

`data/processed/concept_map.csv` supplies KC IDs and their benchmark dimensions.
It is the source of KC identity and dimension metadata.

`mappings/kc_topic_map.csv` and `mappings/skill_taxonomy.csv` carry the
versioned KC-to-skill and KC-to-topic relationships, including review status and
source. The builder validates and preserves existing mapping files. It only
generates a draft when a mapping file is absent.

Topic assignments are metadata only: this build does not aggregate English
learner states into Moodle topic-level states or join the English and Moodle
learner populations. Any such integration requires a separately validated
identity and activity mapping.
