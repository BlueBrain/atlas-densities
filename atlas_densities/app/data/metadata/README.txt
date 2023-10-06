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
