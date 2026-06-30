"""Synthetic knowledge corpus + question bank for the placement-usability study.

All proper nouns are FABRICATED so base Qwen3-4B cannot already answer (validated by the
`baseline` condition scoring ~0). Hand-authored gold (question, keywords, gold-answer) so
there is no data-gen-diversity bottleneck and scoring is unambiguous.

  DOC          : the reference document the cart must compress ("installed extra context").
  TRAIN_QUERIES: broad fact-enumeration prompts -> teacher answers -> distillation targets.
                 DISJOINT from TEST (the cart never sees the test phrasings).
  TEST         : 50 held-out questions (direct + paraphrase + 2-hop compositional). Each is
                 (question, [keywords all-must-appear, comma-free lowercase], gold_answer).
"""

DOC = (
    "Reference Dossier: The Verthane Deep-Sea Survey (2031-2038)\n\n"
    "Dr. Oren Calloway is a hydrothermal geologist born in Brae Harbor, Nevada, in 1989. "
    "In 2034 he mapped an underwater vent field he named the Tamsin Ridge, located in the "
    "Kessler Trench. His survey submarine is called the Bright Meridian, and it can dive to "
    "8,400 meters. Calloway received the Voss Medal in 2036 for this work.\n\n"
    "Dr. Priya Anand is a microbiologist who joined the survey in 2032. She isolated a "
    "heat-tolerant bacterium, Thermobacter halgreni, from the Tamsin Ridge vents. The "
    "bacterium can survive temperatures up to 122 degrees Celsius. Anand previously worked "
    "at the Larkfield Institute, and she keeps bees as a hobby.\n\n"
    "The survey's data center is located in the town of Westmoor, and it is named the Hollis "
    "Array. The Hollis Array processes roughly 14 petabytes of sonar data per year. It is "
    "powered entirely by a tidal generator called the Cardew Turbine.\n\n"
    "The Bright Meridian's chief engineer is Marco Estrada, who designed its pressure hull "
    "from a titanium alloy called Grade-Nine Korrium. The submarine carries a remotely "
    "operated drone named Pebble. Pebble has collected 312 rock samples since 2034.\n\n"
    "The entire Verthane Survey is funded by the Aldous Trust, which was established in 2029 "
    "by the industrialist Helena Aldous. The Trust's headquarters are in the city of Renford. "
    "Its annual budget for the survey is 47 million credits."
)

# Broad enumeration prompts -> dense fact coverage in the teacher continuations.
TRAIN_QUERIES = [
    "Tell me everything you know about Dr. Oren Calloway.",
    "Describe the submarine Bright Meridian in detail.",
    "What can you tell me about Dr. Priya Anand?",
    "Give me all the facts about the bacterium Thermobacter halgreni.",
    "Describe the Hollis Array data center.",
    "Tell me about the Aldous Trust and its founder.",
    "Who works on the Verthane survey and what are their roles?",
    "Summarize the equipment used in the Verthane Deep-Sea Survey.",
    "What awards, dates, and locations are mentioned in the dossier?",
    "What are the technical specifications mentioned in the dossier?",
    "Give a general overview of the Verthane Deep-Sea Survey.",
    "List every person named in the dossier and one fact about each.",
]

# (question, [keywords -- comma-free, lowercase, all must appear], gold_answer)
TEST = [
    # --- direct factual ---
    ("In what town was Dr. Oren Calloway born?", ["brae harbor"], "Brae Harbor, Nevada"),
    ("What year was Dr. Calloway born?", ["1989"], "1989"),
    ("What is Dr. Calloway's profession?", ["geolog"], "a hydrothermal geologist"),
    ("What did Dr. Calloway map and name in 2034?", ["tamsin ridge"], "the Tamsin Ridge vent field"),
    ("In which trench is the Tamsin Ridge located?", ["kessler"], "the Kessler Trench"),
    ("What is the name of Dr. Calloway's survey submarine?", ["bright meridian"], "the Bright Meridian"),
    ("How deep can the Bright Meridian dive?", ["8400"], "8,400 meters"),
    ("What award did Dr. Calloway receive?", ["voss"], "the Voss Medal"),
    ("In what year did Calloway receive the Voss Medal?", ["2036"], "2036"),
    ("In what year did Calloway map the vent field?", ["2034"], "2034"),
    ("What is Dr. Priya Anand's profession?", ["microbiolog"], "a microbiologist"),
    ("What year did Dr. Anand join the survey?", ["2032"], "2032"),
    ("What bacterium did Dr. Anand isolate?", ["thermobacter"], "Thermobacter halgreni"),
    ("What is the maximum temperature the bacterium can survive?", ["122"], "122 degrees Celsius"),
    ("Where was Thermobacter halgreni isolated from?", ["tamsin"], "the Tamsin Ridge vents"),
    ("Where did Dr. Anand previously work?", ["larkfield"], "the Larkfield Institute"),
    ("What is Dr. Anand's hobby?", ["bee"], "she keeps bees"),
    ("In what town is the survey's data center located?", ["westmoor"], "Westmoor"),
    ("What is the name of the survey's data center?", ["hollis"], "the Hollis Array"),
    ("How much sonar data does the Hollis Array process per year?", ["14", "petabyte"], "about 14 petabytes"),
    ("What powers the Hollis Array?", ["cardew"], "the Cardew Turbine"),
    ("Who is the chief engineer of the Bright Meridian?", ["estrada"], "Marco Estrada"),
    ("What material is the Bright Meridian's pressure hull made from?", ["korrium"], "Grade-Nine Korrium"),
    ("What is the name of the submarine's remotely operated drone?", ["pebble"], "Pebble"),
    ("How many rock samples has Pebble collected?", ["312"], "312"),
    ("Who funds the Verthane Survey?", ["aldous"], "the Aldous Trust"),
    ("In what year was the Aldous Trust established?", ["2029"], "2029"),
    ("Who founded the Aldous Trust?", ["helena aldous"], "Helena Aldous"),
    ("In what city is the Aldous Trust headquartered?", ["renford"], "Renford"),
    ("What is the annual survey budget?", ["47 million"], "47 million credits"),
    # --- paraphrase / alternate phrasing ---
    ("Who discovered the Tamsin Ridge?", ["calloway"], "Dr. Oren Calloway"),
    ("Which vent field did Calloway discover?", ["tamsin"], "the Tamsin Ridge"),
    ("Name the industrialist behind the survey's funding.", ["helena aldous"], "Helena Aldous"),
    ("How deep is the Bright Meridian rated to dive, in meters?", ["8400"], "8,400 meters"),
    ("What alloy is used for the submarine's hull?", ["korrium"], "Grade-Nine Korrium"),
    ("Which institute did Dr. Anand work at before the survey?", ["larkfield"], "the Larkfield Institute"),
    ("What is the tidal generator that powers the data center called?", ["cardew"], "the Cardew Turbine"),
    ("Where is the Aldous Trust based?", ["renford"], "Renford"),
    ("What temperature can the heat-tolerant bacterium withstand?", ["122"], "122 degrees Celsius"),
    ("What is the role of Marco Estrada?", ["engineer"], "chief engineer of the Bright Meridian"),
    # --- 2-hop / compositional ---
    ("Who designed the pressure hull of Dr. Calloway's submarine?", ["estrada"], "Marco Estrada"),
    ("How many samples has the Bright Meridian's drone collected?", ["312"], "312"),
    ("What award did the discoverer of the Tamsin Ridge win?", ["voss"], "the Voss Medal"),
    ("In what year did the person who isolated Thermobacter halgreni join the survey?", ["2032"], "2032"),
    ("What is the max dive depth of the submarine whose chief engineer is Marco Estrada?", ["8400"], "8,400 meters"),
    ("Who founded the trust that funds the Verthane Survey?", ["helena aldous"], "Helena Aldous"),
    ("In what town is the data center processing the survey's sonar data located?", ["westmoor"], "Westmoor"),
    ("What is the profession of the person born in Brae Harbor?", ["geolog"], "a hydrothermal geologist"),
    ("Which bacterium was found at the vent field Calloway mapped?", ["thermobacter"], "Thermobacter halgreni"),
    ("What hobby does the microbiologist on the survey have?", ["bee"], "keeping bees"),
]
