The groups_ids.json has the following layout:
    [
      {
        "name": "Cerebellum group",
        "UNION": [
            { "name": "Cerebellum", "with_descendants": true },
            { "name": "arbor vitae", "with_descendants": true }
        ]
      },
      {
        "name": "Molecular layer",
        "INTERSECT": [
          "!Cerebellar group",
          { "name": "@.*molecular layer", "with_descendants": true }
        ]
      },
      {
        "name": "Rest",
        "REMOVE": [
          { "name": "root", "with_descendants": true },
          { "UNION": [ "!Cerebellum group", ] }
        ]
      }
      [....]
    ]

The config is an ordered list of stanzas; each stanza is a dictionary with a "name" key, whose value is the Group Name.
This is followed by a key with one of the following keywords, and a list of clauses:
    * `UNION` keyword creates a union of the ids found by the list of clauses.
    * `INTERSECT` keyword creates an intersection of the ids found by the list of clauses.
    * `REMOVE` keyword removes the ids in the second clause from the those of the first.


A clause is formed of dictionary of the form:
    {"$attribute_name": "$attribute_value", "with_descendants": true}
The attribute name is used for the atlas lookup, things like "name" or "acronym" are valid.

Finally, one can refer to a previous stanza by preceding it with a `!`.

The current `group_ids.json was discussed here:
    https://github.com/BlueBrain/atlas-densities/pull/51#issuecomment-1748445813
    In very short, in older versions of the annotations, fibers and main regions were in separated files (ccfv2 / ccfbbp).
    In the main regions, the space allocated to the fibers were annotated to Basic cell groups and regions.
    For some weird reasons, the fibers do not take the entire space allocated for them.
    Which means that for most of the voxels annotated to Basic cell groups and regions, they correspond to fibers.
    I say most of the voxels because there are also voxels at the frontier between two main regions (e.g. Cerebellum) that were not annotated to their closest leaf region.
    These voxels are gone in ccfv3.
