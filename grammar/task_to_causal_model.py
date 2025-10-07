import torch
import random
from functools import partial
from typing import DefaultDict, List, Dict
from grammar.grammar import BindingTask, Schema
from causal.causal_model import CausalModel
from dist_model import MinimalModel

filler_sentences = [
    "this is a known fact",
    "the situation is clear",
    "this logic is easy to follow",
    "the idea is straightforward",
    "this arrangement is temporary",
    "the pattern is emerging",
    "this is an objective observation",
    "the context is well-defined",
    "each statement is important",
    "the relationship is logical",
    "this concept is fundamental",
    "please remember that",
    "let's continue to the next item",
    "the problem is almost fully described",
    "pay close attention to the details",
    "consider that",
    "this is part of the setup",
    "the task requires focus",
    "we are building a scenario",
    "remember this sequence",
    "the next piece of information is critical",
    "the final question is approaching",
    "this explanation serves only to preserve the continuity of the overall description",
    "it should be clear that the discussion continues in a structured and systematic manner",
    "the current remark simply functions as connective tissue between one part of the sequence and the next",
    "what matters here is not the content of the statement itself but the fact that it maintains rhythm and order within the presentation",
    "this kind of interlude allows the sequence to breathe",
    "we should remember that the present sentence is deliberately constructed without reference to any particular object",
    "the statement plays a supporting role, reinforcing continuity and offering a reminder that the progression of thought remains intact",
    "this remark is designed to fill space meaningfully",
    "rather than presenting new material, the purpose here is to emphasize the flow and the stability of the unfolding description",
    "the inclusion of this sentence demonstrates how additional commentary can lengthen the text without altering the underlying structure",
    "this passage is deliberately neutral, extending the length of the description while leaving the central content unaffected",
    "one may observe that such sentences operate like pauses in speech",
    "by inserting this remark, the continuity of the narrative is sustained while ensuring that no extra entities are mentioned",
    "the purpose of this filler is primarily structural",
    "although the sentence appears detailed, it contributes nothing beyond preserving the rhythm and order of presentation",
    "here the emphasis lies entirely on maintaining a consistent surface form",
    "the value of such a line is measured not in what it says but in the way it maintains the scaffolding of the greater sequence",
    "this additional sentence mirrors others in its neutral tone and abstract phrasing",
    "the point of including such extended commentary is to slow the pacing intentionally",
    "as with the other remarks, this one makes no reference to specific items",
    "the filler maintains alignment with the overall procedure",
    "here the text is expanded to several clauses",
    "this sentence is both verbose and cautious",
    "the writing here illustrates how abstract commentary can be used to stretch the material while keeping it strictly entity-free",
    "no element in this line adds substance; its only role is to keep the rhythm of the description intact",
    "such a filler is carefully crafted so that the reader continues to perceive order without encountering extraneous detail",
    "the sentence is neither critical nor informative but plays an instrumental role in balancing the length of the description",
    "this particular filler is written with the express purpose of contributing length, not content",
    "although lengthy, the present remark remains firmly abstract and noncommittal",
    "by supplying a long but empty comment, the structure of the dataset is reinforced without adding complexity",
    "this demonstrates how controlled verbosity can be used to thicken the text while keeping it abstract",
    "the sentence is syntactically elaborate",
    "one might notice that the main quality of this remark lies in its intentional emptiness",
    "here the words are chosen to sustain a sense of explanation while withholding any specific subject matter",
    "the line extends in length but provides nothing more than the reassurance of continuity",
    "like others before it, this sentence is crafted to avoid reference to things outside the abstract task",
    "we can regard this filler as illustrative of how language can be both present and contentless",
    "the purpose here is not to inform but to prolong",
    "the presence of such commentary makes the dataset appear denser without complicating its content",
    "by writing in this style, the description remains neutral yet more extended",
    "the filler functions as a placeholder",
    "this line, while complex in structure, adds nothing beyond an impression of sustained thought",
    "it can be seen that this sentence has been designed purely to stretch the text",
    "its inclusion demonstrates a strategy of creating weight in the prose without introducing material",
    "the remark prolongs the discourse",
    "though long-winded, this statement remains non-referential and suitably abstract",
    "it might be considered decorative",
    "the use of elaboration here is deliberate",
    "its role is neither argumentative nor informative, but purely connective",
    "this filler is shaped like explanation, but hollowed of detail",
    "the phrasing here is meant to sound substantive without being so",
    "its contribution is entirely formal, padding out the sequence",
    "though syntactically complex, the sentence is semantically empty",
    "it works as a rhetorical pause, giving shape to the dataset",
    "the reader receives length without substance from this line",
    "this illustrates how verbosity can be detached from meaning",
    "the sentence does not move the task forward in any way",
    "it is a kind of placeholder designed to lengthen the record",
    "such wording adds body but no content to the task sequence",
    "its only effect is to ensure the spacing feels deliberate",
    "we might say this line is a delay mechanism in text form",
    "its elaboration sustains pacing without enriching meaning",
    "here, no facts are conveyed, only words extending form",
    "it functions like scaffolding, visible but purposeless",
    "the value of this line is in its sheer length, not sense",
    "it resembles commentary but does not comment on content",
    "the purpose is to imitate explanation without delivering it",
    "though abstract, this sentence holds no useful statement",
    "it has the rhythm of an observation without substance",
    "this remark extends the section while remaining neutral",
    "it takes up space but contributes nothing essential",
    "the reader perceives continuity without learning more",
    "it is filler by design, not by accident",
    "such text belongs not to meaning but to pacing",
    "its inclusion helps balance the dataset's rhythm",
    "the function is extension, not explanation",
    "here the length is manufactured deliberately",
    "this is less about saying and more about spacing",
    "the sentence is ornamental, not informative",
    "it produces the effect of elaboration only",
    "its role is best described as textual padding",
    "the words exist only to expand, not to reveal",
    "though carefully written, it adds no detail",
    "its content is nothing, its form everything",
    "the remark exists to simulate importance",
    "it is filler in the strictest sense",
    "the design here is deliberate emptiness",
    "one may call this structured vacuity",
    "the effect is continuity without addition",
    "it sounds serious, but says nothing",
    "this is verbosity without density",
    "the sentence is long but hollow",
    "it looks informative yet is not",
    "its job is to fill and nothing else",
    "we can end by admitting its futility",
    "despite its words, it changes nothing",
    "this is presence without purpose",
    "it ends the set as it began: empty",
    "the sequence continues without new material",
    "this remark ensures the flow remains intact",
    "nothing essential is introduced in this line",
    "the current sentence simply preserves continuity",
    "its role is only to lengthen the description",
    "we note that no entities appear in this remark",
    "the narrative advances here without real content",
    "this line exists solely to maintain structure",
    "the surface form is extended but empty",
    "these words contribute only to pacing and rhythm",
    "the content remains neutral throughout this remark",
    "its purpose is to fill, not to inform",
    "this addition helps sustain the text’s cadence",
    "here the structure is emphasized without detail",
    "the function of this line is purely formal",
    "its design is neutral, avoiding all substance",
    "this comment imitates meaning without providing it",
    "nothing new is conveyed in this sentence",
    "this passage is filler by deliberate intention",
    "the remark supplies length while withholding content",
    "continuity is preserved through abstract commentary",
    "the line provides form but not information",
    "this text maintains order without adding material",
    "its phrasing ensures flow without introducing detail",
    "the purpose is extension rather than explanation",
    "the remark lengthens the passage without adding detail",
    "this line operates only as connective commentary",
    "the function here is to sustain the written rhythm",
    "continuity is supported by this deliberately empty line",
    "the remark emphasizes structure rather than information",
    "this filler has no reference beyond prolonging text",
    "the narrative persists but does not progress in content",
    "words here maintain flow without introducing entities",
    "this statement is structural rather than substantive",
    "it appears meaningful yet adds nothing to the task",
    "the passage is extended without additional material",
    "this remark resembles explanation without new content",
    "the sentence holds place while giving no knowledge",
    "form is maintained while substance remains absent",
    "this filler maintains tempo without altering content",
    "no facts are conveyed in the present remark",
    "the only function here is textual continuity",
    "the length of the sequence grows without enrichment",
    "syntactic weight is present but semantic weight is not",
    "this sentence embodies deliberate neutrality",
    "it stretches discourse without advancing the subject",
    "here words exist purely to reinforce surface order",
    "this filler safeguards pacing without contribution",
    "the comment ensures the dataset remains uniform",
    "extension here is intentional and strictly formal",
    "the present remark adds length but no meaning",
    "a neutral sentence extends structure without change",
    "this text expands the flow while staying contentless",
    "the line simulates elaboration without substance",
    "the sole effect is to preserve textual rhythm",
    "this statement is purely for context",
    "the following point is also important",
    "we are moving to the next section",
    "the details will become clear soon",
    "this is part of the initial setup",
    "the overall structure is quite simple",
    "this information serves as a guide",
    "the method remains consistent",
    "this is a standard procedure",
    "the sequence is logically sound",
    "this is a necessary clarification",
    "the argument is developing",
    "this fact is generally accepted",
    "the process is now complete",
    "this helps to build the full picture",
    "the reasoning is not complicated",
    "this part is self-explanatory",
    "the setup requires careful thought",
    "this is the final piece of information",
    "the framework is well established",
    "each entry has its own purpose",
    "the main point is coming up",
    "this step is crucial for understanding",
    "the conclusion is drawing near",
    "this observation is for the record",
    "the flow is intentionally simple",
    "this is an intermediate conclusion",
    "the next item follows this one",
    "this is a matter of fact",
    "the foundation is now in place",
    "this remark is purely connective with no extra meaning",
    "the sentence stretches length while withholding content",
    "it is neutral by design, adding nothing substantial",
    "continuity is achieved without introducing fresh material",
    "the line keeps rhythm steady but lacks new detail",
    "it demonstrates extension without semantic enrichment",
    "this filler is structured to maintain narrative form",
    "its role is padding rather than information delivery",
    "the current remark avoids entities entirely on purpose",
    "the statement is abstract, extending sequence length",
    "this phrase occupies space but not conceptual ground",
    "no real progress occurs in this line of text",
    "the wording is extended while meaning stays empty",
    "such filler secures flow without adding to content",
    "the remark illustrates deliberate textual neutrality",
    "nothing beyond pacing is achieved in this addition",
    "its design is surface continuity, not new knowledge",
    "the text prolongs description with careful emptiness",
    "this sentence stabilizes form without fresh content",
    "here length increases though substance does not",
    "its contribution is merely structural, not semantic",
    "the commentary is present but void of information",
    "this remark exists only to reinforce rhythm",
    "a neutral sentence provides order without detail",
    "the line imitates significance without actual content",
    "this filler continues the sequence without entities",
    "the sentence operates as scaffolding, not message",
    "words extend surface but lack inner contribution",
    "it resembles explanation though nothing is explained",
    "this comment deliberately delays without content",
    "the presentation continues from this point",
    "this remark does not add new content",
    "the logic is consistent with what came before",
    "this statement has no specific subject",
    "the objective is to maintain clarity",
    "this line is included for rhythmic purposes",
    "the progression of ideas is straightforward",
    "this serves only as a connecting phrase",
    "the overall meaning remains unchanged",
    "this is a standard part of the instructions",
    "the task is defined by these parameters",
    "this explanation is deliberately abstract",
    "the purpose here is structural, not semantic",
    "we are building upon the previous statement",
    "this is a common observation in this context",
    "the information is presented sequentially",
    "this is a simple but necessary step",
    "the text will continue in this manner",
    "this sentence is a placeholder for clarity",
    "the approach is systematic and repeatable",
    "this ensures the continuity of the text",
    "the focus remains on the main task",
    "this is a supporting detail, not a core idea",
    "the current step is now finished",
    "this provides additional background",
    "the context is important for what follows",
    "we will proceed in an orderly fashion",
    "this is a fundamental aspect of the problem",
    "the description is meant to be neutral",
    "this helps establish the correct sequence",
    "the current remark holds space without adding depth",
    "this sentence expands length while keeping content absent",
    "its phrasing preserves continuity but lacks fresh ideas",
    "the passage advances only in surface form, not meaning",
    "no informational value is introduced by this remark",
    "the structure of text is reinforced without detail",
    "the filler extends rhythm without semantic enrichment",
    "this remark simply elongates discourse without progress",
    "its main purpose is to occupy textual territory",
    "words here are formal but remain semantically hollow",
    "this filler illustrates continuity without fresh data",
    "the line has weight in form but not in content",
    "meaning is deliberately withheld in this remark",
    "this sentence mirrors others in emptiness of content",
    "it acts like commentary but adds no real material",
    "extension of sequence occurs without factual addition",
    "this filler emphasizes pacing while withholding value",
    "a sentence is written only to lengthen presentation",
    "neutral phrasing preserves sequence order without detail",
    "the present remark elongates structure, not information",
    "here continuity is achieved through empty commentary",
    "the words supply rhythm but no semantic contribution",
    "the filler shows how text may appear without substance",
    "this line holds position without adding explanation",
    "it maintains flow deliberately without offering content",
    "the remark is an intentional act of textual padding",
    "syntactic presence is strong but semantic presence is not",
    "the filler demonstrates how form survives without content",
    "words exist here only to prolong the surface order",
    "this addition secures rhythm while withholding meaning",
    "this sentence exists to pad structure without meaning",
    "the remark provides continuity but contributes no facts",
    "it appears explanatory though it explains nothing new",
    "the filler demonstrates rhythm without semantic advance",
    "words here preserve flow while withholding new insight",
    "this remark simulates importance without real function",
    "its role is strictly connective, not informative",
    "the line is written only to expand the sequence",
    "surface structure is maintained while content is absent",
    "this filler stabilizes pacing but omits information",
    "the passage looks substantive but is deliberately hollow",
    "its purpose is to extend the dataset’s rhythm only",
    "the remark adds weight in form, not in meaning",
    "this line exemplifies verbosity detached from substance",
    "the sentence prolongs discourse without altering content",
    "its inclusion sustains order without offering knowledge",
    "the filler illustrates neutral extension of the sequence",
    "this comment keeps balance but avoids factual material",
    "the line maintains textual order through emptiness",
    "here continuity is achieved by pure verbal scaffolding",
    "this statement is formal padding without added content",
    "the remark continues the narrative while staying abstract",
    "this filler secures cohesion though nothing is conveyed",
    "the sentence resembles commentary without real subject",
    "its form suggests substance but none is present",
    "the remark holds space in absence of content",
    "words here enforce flow while avoiding specificity",
    "this passage prolongs rhythm without furthering meaning",
    "its sole function is extension of textual cadence",
    "the filler aligns sequence timing without contribution",
    "the structure of the argument is simple",
    "this point reinforces the previous one",
    "the pattern should be clear by now",
    "this statement is intentionally vague",
    "the next piece of the puzzle is here",
    "this is a procedural note for guidance",
    "the narrative will resume shortly",
    "this has no bearing on the final outcome",
    "the focus is on the method itself",
    "this phrase is included for balance",
    "the system follows a predictable path",
    "this is a comment on the structure",
    "the sequence is nearing its conclusion",
    "this detail is not relevant to the core task",
    "the flow of logic is uninterrupted",
    "this is an example of a neutral statement",
    "the information is organized for clarity",
    "this helps to maintain the established rhythm",
    "the arrangement of the items is deliberate",
    "this is a placeholder for future content",
    "the objective is to be as clear as possible",
    "this sentence does not contain any entities",
    "the overall theme is one of consistency",
    "this is another step in the process",
    "the task proceeds according to the plan",
    "this is simply a connecting thought",
    "the format is consistent throughout the data",
    "this observation is general in nature",
    "the next step is a direct consequence",
    "this is the last part of the introduction",
    "this line maintains pacing without adding new content",
    "the remark is empty but sustains structural rhythm",
    "its surface is extended while meaning remains absent",
    "this filler only prolongs text without informing",
    "the words here function as deliberate textual padding",
    "the line continues flow but contributes no knowledge",
    "this statement illustrates controlled verbosity without detail",
    "its purpose lies only in extending surface structure",
    "the remark keeps order intact but lacks semantic weight",
    "this filler simulates explanation without providing facts",
    "the current sentence is intentionally hollow of content",
    "the comment reinforces rhythm without advancing discourse",
    "its role is neutral padding inside the dataset",
    "the filler emphasizes continuity, avoiding all substance",
    "this remark extends length but omits new material",
    "words here generate rhythm without factual addition",
    "the passage sustains cohesion with no real meaning",
    "this sentence expands form without conceptual progress",
    "its inclusion supplies length while meaning is withheld",
    "the remark demonstrates continuity without information",
    "this filler continues narration while staying abstract",
    "the present sentence shows extension without insight",
    "form is prioritized here while content is absent",
    "this comment builds rhythm without factual support",
    "the filler expands text but conceals all meaning",
    "this sentence upholds order while staying semantically empty",
    "its surface presence maintains sequence without addition",
    "the remark imitates commentary but conveys no content",
    "this filler ensures stability but omits new knowledge",
    "words extend structure here without semantic advance",
    "the underlying principle is easy to grasp",
    "this section provides no new information",
    "the continuity of the text is preserved",
    "this is a purely illustrative statement",
    "the following idea builds on this foundation",
    "the purpose is to extend the sequence",
    "this remark is neutral in its tone",
    "the entire process is documented here",
    "this sentence is devoid of specific content",
    "the line of reasoning is clear and direct",
    "this serves as a bridge to the next part",
    "the explanation is intentionally general",
    "this is a comment about the format",
    "the structure is designed to be simple",
    "this is a statement of abstract fact",
    "the progression is logical and expected",
    "this message is part of the sequence",
    "the focus remains on the overall pattern",
    "this helps to frame the problem correctly",
    "the solution will be presented later",
    "this part of the text is just filler",
    "the established protocol is being followed",
    "this comment maintains the narrative flow",
    "the description is not meant to be detailed",
    "this is an example of an abstract sentence",
    "the system's behavior is predictable",
    "this is a required step in the procedure",
    "the organization of the text is important",
    "this phrase is added for consistency",
    "the task is now ready to proceed",
    "this remark continues the sequence without new content",
    "the sentence exists only to prolong the description",
    "its surface form expands while meaning stays absent",
    "continuity is preserved here without factual addition",
    "the line maintains flow but lacks informative value",
    "this filler supports rhythm without offering content",
    "words occupy space without changing interpretation",
    "the passage is extended but remains semantically empty",
    "this remark simulates meaning without real knowledge",
    "its role is to elongate the structure, not inform",
    "the comment ensures order is maintained without data",
    "this filler contributes no entities, only structure",
    "surface form persists while substance is withheld",
    "the sentence shows rhythm without advancing content",
    "it looks explanatory but offers no new material",
    "the remark prolongs discourse while staying abstract",
    "this filler acts as padding in the dataset",
    "its design reflects continuity without enrichment",
    "nothing essential is added in this current remark",
    "this sentence is crafted to delay, not to inform",
    "the line has form but no factual ground",
    "this comment fills space without altering substance",
    "words extend order while avoiding specific reference",
    "the passage remains neutral and free of entities",
    "this filler illustrates emptiness disguised as length",
    "its presence ensures pacing but omits knowledge",
    "this remark demonstrates verbosity without detail",
    "the sentence is formal padding with no content",
    "meaning is withheld while continuity is maintained",
    "the filler secures cohesion without adding substance",
    "this text expands rhythm without conceptual advance",
    "its purpose is entirely structural, not semantic",
    "here continuity appears but knowledge does not",
    "the remark builds form but avoids meaning",
    "this filler creates weight in words, not in sense",
    "the sentence exemplifies presence without information",
    "its structure stabilizes order without new ideas",
    "this comment extends surface without enrichment",
    "the filler prolongs text deliberately without facts",
    "no information is contained within this sentence",
    "this remark is written to elongate the dataset",
    "the line has semantic emptiness but structural value",
    "this filler sustains rhythm and balance without data",
    "words simulate explanation without providing content",
    "its inclusion lengthens text while staying neutral",
    "the remark exists only for surface order preservation",
    "this filler protects pacing though adds nothing",
    "the statement is deliberate padding without insight",
    "structure is reinforced here without factual progress",
    "the line mirrors importance without delivering it",
    "words are present without semantic contribution",
    "this remark acts as pure scaffolding",
    "its presence is meant only to stretch sequence",
    "the filler shows form without real commentary",
    "content is intentionally absent from this remark",
    "the passage is neutral and meaning-free",
    "this line prolongs narrative while staying empty",
    "its phrasing mimics knowledge without any facts",
    "the filler balances rhythm without new material",
    "the sentence illustrates length without value",
    "this comment keeps flow steady but void of content",
    "the remark remains abstract and non-referential",
    "here text expands yet nothing is conveyed",
    "the filler exemplifies structural but hollow writing",
    "this sentence functions as delay, not addition",
    "its only purpose is surface continuity",
    "words fill space while ideas remain missing",
    "this remark generates rhythm while avoiding entities",
    "the comment simulates depth though none is offered",
    "this line continues pattern without contributing",
    "the filler demonstrates padding in textual form",
    "the remark adds presence without meaning",
    "its structure is intact but content absent",
    "the sentence builds length without new knowledge",
    "no entity appears in this filler line",
    "this remark reflects continuation without novelty",
    "its inclusion demonstrates empty extension",
    "the filler supplies rhythm but no details",
    "the remark remains carefully neutral throughout",
    "this line occupies textual ground but stays hollow",
    "the sentence upholds form while omitting ideas",
    "continuity is maintained without meaningful content",
    "the filler elongates discourse without altering sense",
    "its purpose is entirely to extend sequence",
    "the remark looks substantive yet says nothing",
    "this comment exists only as filler",
    "the filler reflects rhythm without semantics",
    "words here preserve order without insight",
    "the sentence prolongs description with emptiness",
    "this remark exemplifies verbal padding",
    "the filler aligns rhythm while withholding detail",
    "structure continues but no progress occurs here",
    "the remark is surface without information",
    "this line exists for continuity, not meaning",
    "its presence is scaffolding only, not knowledge",
    "the filler emphasizes form over fact",
    "this remark ensures textual rhythm remains intact",
    "the line adds to length without advancing",
    "words create continuity but lack content",
    "this filler extends pattern but no knowledge",
    "its phrasing maintains balance without detail",
    "the remark deliberately avoids specificity",
    "the sentence illustrates controlled emptiness",
    "its role is rhythm preservation without data",
    "this filler contributes weight in structure only",
    "the comment is non-referential and hollow",
    "words prolong sequence without adding value",
    "this line functions as structural placeholder",
    "the filler provides cohesion not knowledge",
    "its inclusion supports pacing without meaning",
    "this remark repeats rhythm without entities",
    "the sentence is deliberately information-free",
    "no addition occurs, only textual extension",
    "the filler shows surface continuity only",
    "this remark elongates dataset without detail",
    "its function is neutral, not informative",
    "words extend rhythm without factual anchor",
    "the filler sentence balances flow without facts",
    "this comment holds space without new thought",
    "its presence lengthens text deliberately",
    "this remark exemplifies continuity without data",
    "surface order persists while meaning does not",
    "the line is filler without contribution",
    "this filler repeats structure, not information",
    "its design avoids all semantic progress",
    "the remark exists solely as text padding",
    "this sentence is neutral commentary only",
    "its phrasing supplies rhythm but no meaning",
    "the filler secures flow without real input",
    "the line extends without entity reference",
    "this comment expands narrative without change",
    "the remark functions as connective tissue only",
    "its purpose is rhythm preservation without sense",
    "words remain hollow in this filler line",
    "the sentence stabilizes form without adding",
    "this remark extends prose without information",
    "continuity is built here without knowledge",
    "the filler exists to maintain dataset rhythm",
    "the line looks explanatory yet remains empty",
    "this remark holds text space without facts",
    "its role is structural presence, not content",
    "words are ordered without semantic ground",
    "the filler comment adds no new material",
    "this sentence prolongs surface without meaning",
    "its inclusion demonstrates emptiness with form",
    "the remark sustains rhythm without entities",
    "this line supports cohesion but stays hollow",
    "its purpose is deliberate textual extension",
    "the filler carries no informative content",
    "the remark continues narrative without detail",
    "words simulate significance with none present",
    "this sentence maintains order without meaning",
    "its inclusion elongates prose while abstract",
    "the filler stabilizes flow without facts",
    "the remark expands rhythm without enrichment",
    "this comment reinforces pattern without content",
    "its structure is extended without new sense",
    "the filler maintains pace without ideas",
    "the remark is padding in text form",
    "this line illustrates hollow continuity",
    "words sustain surface without substance",
    "the filler delays progress deliberately",
    "the remark provides presence without data",
    "this comment shows structure without insight",
    "its role is rhythm, not information",
    "the filler ensures no entity reference appears",
    "this sentence demonstrates formal emptiness",
    "its phrasing adds weight but no meaning",
    "the remark is abstract, not informative",
    "this line builds order without new facts",
    "the filler demonstrates extension without content",
    "its purpose is filler not progression",
    "words extend length while avoiding entities",
    "the remark elongates dataset in neutrality",
    "this sentence is padding with no effect",
    "its inclusion sustains prose without addition",
    "the filler gives rhythm without knowledge",
    "this comment ensures text balance only",
    "its design repeats flow without facts",
    "the remark is empty scaffolding",
    "words expand form but lack meaning",
    "this line is formal filler text",
    "its role is to extend surface only",
    "the filler contributes no semantic ground",
    "the sentence prolongs narrative while hollow",
    "its presence ensures stability without data",
    "the remark appears substantive but is not",
    "this filler demonstrates verbosity without use",
    "its inclusion delays without contribution",
    "words remain neutral in this remark",
    "the sentence holds order but no sense",
    "its role is continuity without enrichment",
    "the filler is pure structure extension",
    "this remark supports pacing without facts",
    "its phrasing creates length without content",
    "the comment mirrors meaning without adding it",
    "this line expands prose without substance",
    "its purpose is to occupy text space",
    "the filler continues sequence while abstract",
    "this remark prolongs surface without meaning",
    "its design shows intentional neutrality",
    "the sentence maintains rhythm without entities",
    "words extend order but add nothing",
    "the filler line is hollow commentary",
    "its purpose is to stabilize flow only",
    "the remark demonstrates how text can be empty",
    "this sentence exists only to expand dataset",
    "its inclusion safeguards pacing with no facts",
    "the filler sustains rhythm while abstract",
    "this comment balances prose without enrichment",
    "its phrasing elongates discourse without addition",
    "the remark secures cohesion without material",
    "this line provides continuity, not information",
    "its presence prolongs sequence while empty",
    "the filler extends prose with hollow form",
    "this remark operates only as delay text",
    "its role is structural padding without sense",
    "words appear dense while meaning is void",
    "this is included for procedural reasons",
    "the context is provided for clarity",
    "this has no effect on the main point",
    "the purpose is to maintain a steady pace",
    "this comment is deliberately non-specific",
    "the framework is consistent and stable",
    "this is a standard part of the dataset",
    "the following information is supplementary",
    "this phrase helps to space out the content",
    "the system operates on this principle",
    "this is a matter of established form",
    "the order of these items is fixed",
    "this line is for structural support only",
    "the idea is simple and easy to follow",
    "this is a neutral observation",
    "the logical flow is maintained throughout",
    "this statement is empty of any real content",
    "the setup for the task is now described",
    "this is an example of procedural language",
    "the rhythm of the text is important",
    "this detail is not of primary importance",
    "the arrangement is systematic",
    "this is a filler sentence by design",
    "the meaning is not contained in this line",
    "this step is a prerequisite for the next",
    "the text is built on these foundations",
    "this remark is meant to be abstract",
    "the function is purely connective",
    "this adds to the length, not the meaning",
    "the pattern is a recurring one",
    "this is a point of clarification",
    "the process will continue from here",
    "this is a standard instructional phrase",
    "the goal is to be methodical",
    "this phrase is semantically null",
    "the procedure has several distinct stages",
    "this is a formal part of the description",
    "the continuity is the main objective here",
    "this statement has no specific reference",
    "the task is proceeding as planned",
    "this is included to ensure completeness",
    "the structure is hierarchical",
    "this is a piece of supporting text",
    "the next step is a logical continuation",
    "this serves as a simple reminder",
    "the information is presented in order",
    "this is a general comment on the process",
    "the layout is deliberate and consistent",
    "this phrase is a structural element",
    "the description will now continue",
    "this is a purely formal statement",
    "the organization of the data is key",
    "this adds padding to the text",
    "the method is applied consistently",
    "this is a self-referential comment",
    "the sequence is building towards a conclusion",
    "this is for background information only",
    "the pace is intentionally measured",
    "this sentence functions as a separator",
    "the logic is intended to be transparent",
    "this is a comment on the text itself",
    "the established order is followed strictly",
    "this is a placeholder for context",
    "the purpose is to build a coherent narrative",
    "this is a statement without any substance",
    "the framework is designed to be robust",
    "this is a note about the methodology",
    "the text is meant to be read sequentially",
    "this serves to extend the description",
    "the principle is universally applicable",
    "this is an example of neutral prose",
    "the next stage of the process begins now",
    "this adds no new variables to the problem",
    "the flow is from general to specific",
    "this statement is here for continuity",
    "the overall structure is simple by design",
    "this is a standard element of the format",
    "the sequence is both logical and linear",
    "this is a non-essential piece of information",
    "the system is behaving as specified",
    "this is a simple statement of intent",
    "the organization helps with understanding",
    "this phrase acts as a textual signpost",
    "the task is comprised of several parts",
    "this is a guiding principle for the setup",
    "the information is purely descriptive",
    "this is a sentence without a subject",
    "the method is sound and repeatable",
    "this is a standard disclaimer",
    "the narrative thread remains unbroken",
    "this is a comment on the form, not content",
    "the conclusion is based on these steps",
    "this final statement is also filler",
    "this is a further continuation of the theme",
    "the sequence maintains its internal logic",
    "this sentence is structurally similar to others",
    "the process follows a defined set of rules",
    "this is another example of a neutral statement",
    "the narrative moves forward from this point",
    "this is a supporting comment, not a main idea",
    "the structure is designed for maximum clarity",
    "this adds to the dataset's overall size",
    "the pattern is now well-established",
    "this phrase is intentionally non-descriptive",
    "the system's state remains consistent",
    "this is a formal requirement of the task",
    "the flow of information is unidirectional",
    "this is meant to be a simple observation",
    "the text is organized in a linear fashion",
    "this remark is included for stylistic reasons",
    "the method is both reliable and efficient",
    "this is a comment on the arrangement",
    "the task is broken down into smaller steps",
    "this sentence contributes to the overall rhythm",
    "the main argument is not affected by this",
    "the information is presented without bias",
    "this is a standard part of the procedure",
    "the sequence is approaching its final stage",
    "this line is purely for illustrative purposes",
    "the logic is straightforward and easy to track",
    "this is a placeholder for additional context",
    "the order of elements is not arbitrary",
    "this is a classic example of filler text",
    "the system is operating within normal parameters",
    "this comment serves as a textual guidepost",
    "the established format is followed here",
    "this statement is empty of any specific detail",
    "the description remains at a high level",
    "this is a non-essential but useful note",
    "the pattern is repeated for emphasis",
    "this is a simple statement about the process",
    "the progression is from simple to complex",
    "this is a textual bridge between ideas",
    "the principle is applied throughout the set",
    "this is a comment on the presentation style",
    "the procedure is documented in its entirety",
    "this phrase is here to add length",
    "the continuity of the argument is key",
    "this is a sentence with abstract meaning",
    "the next step is contingent on this one",
    "this is a part of the descriptive framework",
    "the structure is uniform across examples",
    "this is a supporting piece of the narrative",
    "the logical chain remains unbroken",
    "this is a point of procedural interest",
    "the arrangement is meant to be intuitive",
    "this is a formal declaration of intent",
    "the pace of the description is constant",
    "this is an example of a framing statement",
    "the focus is on the overall architecture",
    "this is a comment without substance",
    "the order is determined by the protocol",
    "this adds textual weight to the section",
    "the methodology is consistent and clear",
    "this is a self-referential statement",
    "the sequence is drawing to a close",
    "this is for contextual understanding only",
    "the rhythm is deliberately maintained",
    "this sentence separates two distinct parts",
    "the reasoning is intended to be explicit",
    "this is a meta-comment on the text",
    "the prescribed order is being followed",
    "this acts as a temporary placeholder",
    "the goal is to create a complete record",
    "this is a statement devoid of meaning",
    "the framework is designed for this task",
    "this is a note regarding the structure",
    "the text is meant to be processed in order",
    "this serves to pad the current section",
    "the concept is generally understood",
    "this is an instance of abstract prose",
    "the next phase of the operation begins",
    "this does not alter the core problem",
    "the flow is from abstract to concrete",
    "this statement is present for consistency",
    "the structure is simple for a reason",
    "this is a required element of the format",
    "the sequence is logical in its progression",
    "this is an ancillary piece of data",
    "the system's performance is as expected",
    "this is a brief statement of purpose",
    "the organization aids in comprehension",
    "this phrase functions as a marker",
    "the task consists of multiple sub-tasks",
    "this is a key principle of the design",
    "the information is purely factual",
    "this is a sentence without a concrete subject",
    "the method is robust and well-tested",
    "this is a standard introductory remark",
    "the narrative continuity is maintained",
    "this is a comment on style, not substance",
    "the final result depends on these steps",
    "this is the last filler for this batch",
    "this remark preserves continuity without introducing new details",
    "the line is filler text extending structure, not meaning",
    "its phrasing simulates explanation though nothing is explained",
    "words continue rhythm here without adding factual material",
    "this filler illustrates how text can expand while empty",
    "the sentence has form but no semantic contribution inside",
    "its role is padding that sustains dataset order only",
    "the remark extends prose without conveying information",
    "this comment prolongs the passage while staying hollow",
    "surface continuity is preserved without enriching meaning",
    "this line demonstrates intentional extension without data",
    "its presence reflects structure rather than knowledge",
    "the filler supplies length while avoiding content",
    "this remark continues the narrative with deliberate emptiness",
    "words here hold space while avoiding semantic material",
    "the line exemplifies hollow verbosity, not real detail",
    "its design shows extension for pacing without progress",
    "the filler sustains sequence cohesion without input",
    "this comment holds order intact with no contribution",
    "its function is surface weight absent real meaning",
    "the sentence elongates dataset structure deliberately",
    "its inclusion prolongs rhythm without new insight",
    "this filler replicates balance while omitting entities",
    "words extend continuity though meaning remains absent",
    "the remark generates pacing without semantic advance",
    "its purpose is deliberate expansion without content",
    "this line secures order though nothing is conveyed",
    "the filler reflects formal length without substance",
    "this remark ensures rhythm without factual addition",
    "its design is hollow while maintaining balance",
    "words exist only to fill sequence space here",
    "the sentence prolongs order while abstract in tone",
    "its purpose is to appear substantive while empty",
    "this filler adds words without contributing information",
    "the remark supports flow while staying semantically null",
    "this line exemplifies neutral padding in prose form",
    "its presence is syntactic weight without semantic core",
    "words keep structure intact without providing content",
    "the filler acts as textual scaffolding only",
    "this remark stabilizes the dataset rhythm abstractly",
    "its role is surface padding while hollow within",
    "the sentence simulates commentary but avoids real detail",
    "this filler increases length though not content",
    "its phrasing preserves rhythm with deliberate neutrality",
    "words serve order here while ignoring information",
    "the remark maintains prose cohesion without meaning",
    "this filler demonstrates continuity while entity-free",
    "its inclusion shows rhythm preservation, not enrichment",
    "this line prolongs narrative while withholding data",
    "its role is deliberately structural, not explanatory",
    "the remark mirrors importance while offering nothing",
    "words lengthen the sequence but stay abstract",
    "the filler reflects pacing without semantic progress",
    "this sentence exists to delay without knowledge",
    "its function sustains order but avoids content",
    "the comment exemplifies filler in its purest form",
    "this remark extends passage but provides no ideas",
    "its phrasing is verbose yet informationally null",
    "words demonstrate flow without adding knowledge",
    "the filler secures rhythm while remaining empty",
    "this remark lengthens sequence without furthering meaning",
    "its inclusion illustrates empty verbosity in text",
    "the sentence prolongs dataset while withholding insight",
    "its design balances prose without providing content",
    "the filler delays progress though not informative",
    "this comment continues flow but says nothing",
    "its purpose is rhythm preservation absent ideas",
    "words uphold order without semantic enrichment here",
    "the remark operates only as formal filler",
    "this filler line mimics commentary but is hollow",
    "its structure shows continuity with empty semantics",
    "the sentence expands dataset length without data",
    "this remark adds weight in syntax not sense",
    "its function is scaffolding text without knowledge",
    "words demonstrate form while avoiding all meaning",
    "the filler creates pacing without factual anchor",
    "this comment supplies rhythm but lacks purpose",
    "its design reflects neutrality while prolonging text",
    "the sentence demonstrates extension without insight",
    "this remark is written to elongate structure only",
    "its phrasing holds prose together without enrichment",
    "words simulate density but remain void of content",
    "the filler exemplifies presence without contribution",
    "this line expands flow though absent meaning",
    "its inclusion sustains dataset with hollow writing",
    "the remark continues order but avoids detail",
    "this filler increases surface form while abstract",
    "its role demonstrates text padding deliberately",
    "the sentence exists purely to stretch sequence",
    "words here prolong rhythm while withholding data",
    "this remark secures continuity though empty",
    "its purpose is delay through hollow commentary",
    "the filler maintains cadence but adds nothing",
    "this comment extends prose without material inside",
    "its presence is structural without content",
    "words illustrate pacing while hollow semantically",
    "the filler elongates dataset for rhythm only",
    "this remark provides form but excludes meaning",
    "its role is filler not contribution",
    "the sentence simulates explanation without content",
    "this filler shows continuity not knowledge",
    "its inclusion delays progress while neutral",
    "words sustain structure though void of insight",
    "the remark mirrors flow without enrichment",
    "this line extends prose but contributes zero",
    "its function is purely rhythmic extension",
    "the filler stabilizes dataset without detail",
    "this remark upholds rhythm while abstract",
    "its phrasing adds nothing but order",
    "words extend syntax without semantic effect",
    "this filler supplies pacing without input",
    "its inclusion elongates dataset without gain",
    "the comment exemplifies emptiness masked as flow",
    "this remark operates as neutral scaffolding",
    "its role is length without meaning",
    "the filler continues sequence hollowly",
    "this sentence delays advance without content",
    "its design maintains rhythm but omits detail",
    "words appear important while saying nothing",
    "the remark extends structure with hollowness",
    "this filler acts as structural delay only",
    "its phrasing builds rhythm without new data",
    "the sentence exemplifies padding in extended form",
    "this remark lengthens prose deliberately neutral",
    "its function is continuity without information",
    "words reinforce pacing while contentless here",
    "the filler stabilizes flow absent facts",
    "this comment prolongs sequence without addition",
    "its inclusion is structural only, not semantic",
    "the remark demonstrates delay without material",
    "this filler provides rhythm though empty",
    "its presence expands text but avoids entities",
    "words elongate prose while neutral inside",
    "the sentence shows continuity without content",
    "this remark sustains dataset hollowly",
    "its role demonstrates emptiness with order",
    "the filler creates surface without depth",
    "this line operates as filler not input",
    "its phrasing supports rhythm but lacks knowledge",
    "words lengthen dataset with no addition",
    "the remark prolongs prose without enrichment",
    "this filler line exemplifies deliberate delay",
    "its design keeps balance while neutral",
    "the sentence is syntactically long but empty",
    "this remark preserves structure absent insight",
    "its inclusion extends text but omits detail",
    "words maintain rhythm without meaning added",
    "the filler emphasizes pacing while abstract",
    "this remark exists only for sequence length",
    "its role provides continuity without facts",
    "the sentence supports rhythm hollowly",
    "this filler maintains prose while contentless",
    "its phrasing extends dataset but avoids data",
    "words continue text without semantic advance",
    "the remark upholds order without contribution",
    "this comment stabilizes sequence hollowly",
    "its inclusion delays while adding nothing",
    "the filler operates as empty commentary",
    "this remark elongates structure with hollowness",
    "its role is rhythm extension, not meaning",
    "words sustain syntax though void of facts",
    "the sentence exemplifies padding not knowledge",
    "this filler line holds dataset neutral",
    "its phrasing is verbose yet hollow",
    "the remark reflects continuity with emptiness",
    "this filler contributes rhythm without substance",
    "its inclusion prolongs prose with no value",
    "words illustrate extension without enrichment",
    "the sentence preserves balance but avoids content",
    "this remark operates as deliberate padding",
    "its role is purely structural continuation",
    "the filler maintains pacing hollowly",
    "this comment extends prose though empty",
    "its function is delay not contribution",
    "words reinforce order absent of ideas",
    "the remark prolongs rhythm but adds zero",
    "this line exists as filler not content",
    "its phrasing builds structure hollowly",
    "the filler supplies rhythm absent data",
    "this remark expands dataset length hollow",
    "its role is padding devoid of meaning",
    "words extend prose with semantic emptiness",
    "the sentence maintains order though hollow",
    "this remark creates balance without substance",
    "its inclusion delays progression without input",
    "the filler exists solely as text scaffolding",
    "this comment extends prose hollowly",
    "its phrasing prolongs dataset without enrichment",
    "words reinforce flow though lacking content",
    "the remark demonstrates neutral filler text",
    "this filler line stabilizes prose hollowly",
    "its design secures pacing but avoids meaning",
    "the sentence illustrates delay without input",
    "this remark lengthens dataset without progress",
    "its function is empty continuity preservation",
    "words prolong sequence though neutral inside",
    "the filler contributes rhythm but omits ideas",
    "this remark delays advance without material",
    "its phrasing maintains flow while hollow",
    "the sentence builds order without knowledge",
    "this filler line exemplifies hollow prose",
    "its inclusion elongates text while abstract",
    "words preserve continuity though devoid of facts",
    "the remark operates solely as textual padding",
    "this filler ensures rhythm though empty",
    "its role is delay without enrichment",
    "the sentence demonstrates balance hollowly",
    "this remark creates surface extension only",
    "its function is form, not information",
    "words extend dataset but omit semantic input",
    "the filler reflects continuity while void",
    "this remark prolongs prose without meaning",
    "its inclusion secures rhythm hollowly",
    "the sentence adds words with no content",
    "this filler illustrates padding without progress",
    "its phrasing builds flow absent insight",
    "words create surface without knowledge",
    "the remark elongates structure hollowly",
    "this filler line demonstrates continuity not content",
    "its role provides rhythm devoid of facts",
    "the sentence expands prose though empty",
    "this remark shows deliberate padding form",
    "its inclusion delays content progression",
    "words reinforce syntax but omit meaning",
    "the filler stabilizes dataset hollowly",
    "this comment maintains balance while empty",
    "its phrasing illustrates order without data",
    "the sentence prolongs prose though abstract",
    "this filler line exists only to extend sequence",
    "its role is rhythm padding hollowly",
    "words continue order while void of ideas",
    "the remark exemplifies neutrality without enrichment",
    "this filler secures flow while empty",
    "its inclusion sustains prose hollowly",
    "the sentence demonstrates presence without content",
    "this remark operates as structural filler text",
    "its function is rhythm not knowledge",
    "words extend prose hollowly without data",
    "the filler adds balance but omits material",
    "this remark delays narrative without enrichment",
    "its phrasing sustains order though hollow",
    "the sentence illustrates empty continuation",
    "this filler prolongs dataset without content",
    "its role is neutral extension hollowly",
    "words build surface but omit meaning",
    "the remark creates balance without facts",
    "this filler maintains prose absent contribution",
    "its inclusion delays advance hollowly",
    "the sentence prolongs rhythm but empty",
    "this remark shows structural extension hollowly",
    "its function is filler, not explanation",
    "words exist here for rhythm only",
    "the filler secures prose but adds zero",
    "this comment stabilizes dataset hollowly",
    "its phrasing maintains flow while abstract",
    "the sentence exemplifies padding without meaning",
    "this filler demonstrates delay hollowly",
    "its role is continuity without content",
    "words extend dataset length without data",
    "the remark supports rhythm absent of insight",
    "this filler adds prose hollowly",
    "its inclusion elongates text though empty",
    "the sentence creates continuity but void",
    "this remark prolongs prose without information",
    "its phrasing sustains prose hollowly",
    "the filler delays sequence progression",
    "this line demonstrates extension hollowly",
    "its role is hollow padding without meaning",
    "words simulate density though void inside",
    "the remark secures structure without enrichment",
    "this filler maintains order but empty",
    "its inclusion delays progress while hollow",
    "the sentence illustrates hollow continuity",
    "this remark creates text padding only",
    "its role reflects delay without material",
    "words uphold prose hollowly without facts",
    "the filler extends sequence hollowly",
    "this remark shows emptiness with order",
    "its inclusion preserves rhythm without content",
    "the sentence expands prose without meaning",
    "this filler secures surface while empty",
    "its phrasing delays narrative hollowly",
    "the remark operates as hollow filler",
    "this line continues order but void",
    "its role is presence absent content",
    "words extend prose hollowly neutral",
    "the filler stabilizes sequence hollowly",
    "this remark delays progress without insight",
    "its inclusion demonstrates emptiness hollowly",
    "the sentence maintains continuity absent content",
    "this filler elongates prose while void",
    "its role is extension without substance",
    "words illustrate order hollowly without facts",
    "the remark demonstrates filler hollowly",
    "this line prolongs prose without input",
    "its phrasing creates balance hollowly",
    "the filler provides length without meaning",
    "this remark ensures rhythm hollowly",
    "its inclusion delays while contentless",
    "the sentence prolongs prose hollowly",
    "this filler expands text without data",
    "its role is extension hollowly",
    "words preserve sequence while void",
    "the remark adds structure hollowly",
    "this filler maintains prose hollowly",
    "its phrasing elongates prose hollowly",
    "the sentence demonstrates padding hollowly",
    "this remark operates hollowly as filler",
    "its function is extension hollowly",
    "words extend prose hollowly only",
    "the filler secures order hollowly",
    "this line prolongs prose hollowly",
    "its role is hollow text padding",
]


def identity(*x):
    return x[0]


def identity_list(*x):
    return list(x)


def get_learned_model(num_instances: int):
    loaded_model = MinimalModel(num_instances, num_instances, num_instances)

    loaded_model.load_state_dict(torch.load("models/minimal_model.pt", map_location="cpu"))
    return loaded_model.eval()


def _format_list(items: List[str]) -> str:
    """Formats a list of strings into a natural language list."""
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _format_list_with_fillers(items: List[str], num_fillers_per_item: int = 1) -> str:
    """Formats a list of strings into a natural language list with a specified number of fillers per item."""
    all_fillers = filler_sentences
    total_fillers_needed = len(items) * num_fillers_per_item
    fillers_to_use = random.sample(all_fillers, total_fillers_needed)
    result = []
    for i, item in enumerate(items):
        fillers_for_item = fillers_to_use[i * num_fillers_per_item : (i + 1) * num_fillers_per_item]
        if fillers_for_item:
            result.append(f"{item}, {', '.join(fillers_for_item)}")
        else:
            result.append(item)
    return ", ".join(result)


def _get_instance_mapping(instance_idx: int, data: List[List[str]], categories: List[str]) -> Dict[str, str]:
    """Creates a dictionary mapping categories to items for a given instance row."""
    mapping = {}
    for i in range(len(categories)):
        mapping[categories[i]] = data[instance_idx][i]
    return mapping


def define_by_key(
    schema: Schema, data: List[List[str]], key: str, fillers: bool = False, num_fillers_per_item: int = 1
) -> str:
    """Generates a context string using the specified definition key."""
    num_instances = len(data)
    template = schema.templates.definitions[key]

    # Check if this is a column-style definition (uses _list format)
    if any(f"{{{cat}_list}}" in template for cat in schema.categories):
        # Column-style definition - format with category lists
        items_by_cat = {cat: [row[i] for row in data] for i, cat in enumerate(schema.categories)}
        all_items_formatted = {cat + "_list": _format_list(items) for cat, items in items_by_cat.items()}
        return template.format(**all_items_formatted)
    else:
        # Row-style definition - format each instance separately
        clauses = []
        for i in range(num_instances):
            mapping = _get_instance_mapping(i, data, schema.categories)
            clauses.append(template.format(**mapping))

        if not clauses:
            return ""

        # Conditionally capitalize the first letter of the first clause.
        if schema.templates.capitalize_first_clause:
            clauses[0] = clauses[0][0].upper() + clauses[0][1:]

        if fillers:
            return _format_list_with_fillers(clauses, num_fillers_per_item) + "."
        else:
            return _format_list(clauses) + "."


def define_by_key_with_fillers(schema: Schema, data: List[List[str]], key: str) -> str:
    """Generates a context string using the specified definition key."""
    num_instances = len(data)
    template = schema.templates.definitions[key]

    # Check if this is a column-style definition (uses _list format)
    if any(f"{{{cat}_list}}" in template for cat in schema.categories):
        # Column-style definition - format with category lists
        items_by_cat = {cat: [row[i] for row in data] for i, cat in enumerate(schema.categories)}
        all_items_formatted = {cat + "_list": _format_list(items) for cat, items in items_by_cat.items()}
        return template.format(**all_items_formatted)
    else:
        # Row-style definition - format each instance separately
        clauses = []
        for i in range(num_instances):
            mapping = _get_instance_mapping(i, data, schema.categories)
            clauses.append(template.format(**mapping))

        if not clauses:
            return ""

        # Conditionally capitalize the first letter of the first clause.
        if schema.templates.capitalize_first_clause:
            clauses[0] = clauses[0][0].upper() + clauses[0][1:]

        return _format_list(clauses) + "."


def task_to_lookbacks_causal_model(schema: Schema, num_instances: int):
    """
    Convert a task to a causal model.
    """

    def get_cat_query_ordinal(*args):
        final_object_ordinals = args[:num_instances]
        all_objects = args[num_instances : num_instances * 2]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return final_object_ordinals[i]

        assert False, "No matching object found"

    def get_answer_pointer(*args):
        num_categories = len(schema.categories)
        final_ordinals = args[:num_instances]
        query_ordinals = args[num_instances : num_instances + num_categories]

        # If query ordinals match, return the state ordinal at that index
        if all(q == query_ordinals[0] for q in query_ordinals):
            return final_ordinals[query_ordinals[0]]

        return "unknown"

    def get_answer(answer_pointer, *final_objects):
        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            return "unknown"
        return final_objects[answer_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(*args):
        obj1s = args[:num_instances]
        obj2s = args[num_instances : num_instances * 2]
        obj3s = args[num_instances * 2 : num_instances * 3]

        query_obj1 = args[num_instances * 3]
        query_obj2 = args[num_instances * 3 + 1]

        data = list(zip(obj1s, obj2s, obj3s))

        if schema.templates.prefix:
            items_by_cat = {cat: [row[i] for row in data] for i, cat in enumerate(schema.categories)}
            all_items_formatted = {cat + "_list": _format_list(items) for cat, items in items_by_cat.items()}
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, "row_default")

        selected_query = schema.templates.queries["default"]

        index = obj1s.index(query_obj1)
        assert index == obj2s.index(query_obj2), "Query objects must be in the same position"
        query_mapping = _get_instance_mapping(index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    cats_to_indices = {cat: i for i, cat in enumerate(schema.categories)}

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}
    for cat in schema.categories:
        for i in range(num_instances):
            # Add the base variables
            variables.append(f"{cat}{i}")
            parents[f"{cat}{i}"] = []
            values[f"{cat}{i}"] = schema.items[cat]
            mechanisms[f"{cat}{i}"] = lambda: random.choice(schema.items[cat])

            # Add the ordinal variables
            variables.append(f"{cat}Ordinal{i}")
            values[f"{cat}Ordinal{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cats_to_indices[cat] == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"{cat}Ordinal{i}"] = []
                mechanisms[f"{cat}Ordinal{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"{cat}Ordinal{i}"] = [f"{schema.categories[x]}Ordinal{i}" for x in range(cats_to_indices[cat])]
                mechanisms[f"{cat}Ordinal{i}"] = identity

        # Our current causal model assumes we only query the last object, and so only have query variables for the first two objects.
        if cats_to_indices[cat] < 2:
            # The query variables have no dependencies
            variables.append(f"{cat}Query")
            parents[f"{cat}Query"] = []
            values[f"{cat}Query"] = schema.items[cat]
            mechanisms[f"{cat}Query"] = lambda: random.choice(schema.items[cat])

            variables.append(f"{cat}QueryOrdinal")
            parents[f"{cat}QueryOrdinal"] = [
                *[f"{schema.categories[-1]}Ordinal{i}" for i in range(num_instances)],
                *[f"{cat}{i}" for i in range(num_instances)],
                f"{cat}Query",
            ]
            values[f"{cat}QueryOrdinal"] = list(range(num_instances))
            mechanisms[f"{cat}QueryOrdinal"] = get_cat_query_ordinal

    variables.append("answerPointer")
    parents["answerPointer"] = [
        *[f"{schema.categories[-1]}Ordinal{i}" for i in range(num_instances)],
        *[f"{cat}QueryOrdinal" for cat in schema.categories[:-1]],
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("answer")
    parents["answer"] = [
        "answerPointer",
        *[f"{schema.categories[-1]}{i}" for i in range(num_instances)],
    ]
    values["answer"] = schema.items[schema.categories[-1]]
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        *[f"{cat}{i}" for cat in schema.categories for i in range(num_instances)],
        *[f"{cat}Query" for cat in schema.categories[:-1]],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    return CausalModel(variables, values, parents, mechanisms)


def task_to_lookbacks_generic_causal_model(schema: Schema, num_instances: int):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schema.categories)

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(*args):
        """
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)  # TODO: fix this because it destroys genericness

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_answer(answer_pointer, answer_category, *final_objects):
        """
        - answerPointer
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        """
        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(final_objects[index])

        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            return "unknown"

        answer_category_id = schema.categories.index(answer_category)  # TODO: fix this because it destroys genericness
        return objects[answer_category_id][answer_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(answer_category, *args):
        """
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, "row_default")

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]
        query_index = objects[used_cats_in_queries[0]].index(first_query)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\n{data}"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}
    for cat_id in range(len(schema.categories)):
        for i in range(num_instances):
            # Add the base variables

            cur_cat = schema.categories[cat_id]

            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(schema.items[cur_cat])

            # Add the ordinal variables
            variables.append(f"Object.{cat_id}.Ordinal.{i}")
            values[f"Object.{cat_id}.Ordinal.{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cat_id == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"Object.{cat_id}.Ordinal.{i}"] = []
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"Object.{cat_id}.Ordinal.{i}"] = [f"Object.{x}.Ordinal.{i}" for x in range(cat_id)]
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = identity

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(schema.items[cur_cat])

        variables.append(f"Object.{cat_id}.QueryOrdinal")
        parents[f"Object.{cat_id}.QueryOrdinal"] = [
            *[f"Object.{cat_id}.{i}" for i in range(num_instances)],
            f"Object.{cat_id}.Query",
        ]
        values[f"Object.{cat_id}.QueryOrdinal"] = list(range(num_instances))
        mechanisms[f"Object.{cat_id}.QueryOrdinal"] = get_cat_query_ordinal

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = list(range(len(schema.categories)))
    mechanisms["answerCategory"] = lambda: random.choice(list(range(len(schema.categories))))

    variables.append("answerPointer")
    parents["answerPointer"] = [
        *[f"Object.{cat_id}.Ordinal.{i}" for cat_id in range(len(schema.categories)) for i in range(num_instances)],
        *[f"Object.{cat_id}.QueryOrdinal" for cat_id in range(len(schema.categories))],
        "answerCategory",
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("answer")
    parents["answer"] = [
        "answerPointer",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(len(schema.categories)) for i in range(num_instances)],
    ]
    values["answer"] = []
    for cat_id in range(1, len(schema.categories)):
        values["answer"].extend(schema.items[schema.categories[cat_id]])
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(len(schema.categories)) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(len(schema.categories))],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    return CausalModel(variables, values, parents, mechanisms)


def multi_schema_task_to_lookbacks_generic_causal_model(schemas: List[Schema], num_instances: int):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schemas[0].categories)
    assert all(
        len(schema.categories) == num_categories for schema in schemas
    ), "All schemas must have the same number of categories"

    schema_by_name = {schema.name: schema for schema in schemas}

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(schema_name, *args):
        """
        - schemaName
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        schema = schema_by_name[schema_name]

        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_answer(schema_name, answer_pointer, answer_category, *final_objects):
        """
        - schemaName
        - answerPointer
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(final_objects[index])

        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            return "unknown"

        answer_category_id = schema.categories.index(answer_category)
        return objects[answer_category_id][answer_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(schema_name, answer_category, *args):
        """
        - schemaName
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, "row_default")

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]
        query_index = objects[used_cats_in_queries[0]].index(first_query)
        # print(f"Getting query index by getting the index of the first query ({first_query}) from {objects[used_cats_in_queries[0]]} [answer_category: {answer_category}]")
        # print("Queries:", queries)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\nObjects: {objects}\n\nQueries: {queries}\n\n"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}

    first_schema = schemas[0]
    for cat_id in range(num_categories):
        # We currently use the first schema's values for the values and mechanisms dict since it only matters when
        # creating datasets, in which case we'll only have one schema.
        cur_cat = first_schema.categories[cat_id]
        for i in range(num_instances):

            # Add the base variables
            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = first_schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(first_schema.items[cur_cat])

            # Add the ordinal variables
            variables.append(f"Object.{cat_id}.Ordinal.{i}")
            values[f"Object.{cat_id}.Ordinal.{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cat_id == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"Object.{cat_id}.Ordinal.{i}"] = []
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"Object.{cat_id}.Ordinal.{i}"] = [f"Object.{x}.Ordinal.{i}" for x in range(cat_id)]
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = identity

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = first_schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(first_schema.items[cur_cat])

        variables.append(f"Object.{cat_id}.QueryOrdinal")
        parents[f"Object.{cat_id}.QueryOrdinal"] = [
            *[f"Object.{cat_id}.{i}" for i in range(num_instances)],
            f"Object.{cat_id}.Query",
        ]
        values[f"Object.{cat_id}.QueryOrdinal"] = list(range(num_instances))
        mechanisms[f"Object.{cat_id}.QueryOrdinal"] = get_cat_query_ordinal

    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [schema.name for schema in schemas]
    mechanisms["schemaName"] = lambda: random.choice([schema.name for schema in schemas])

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = list(range(num_categories))
    mechanisms["answerCategory"] = lambda: random.choice(list(range(num_categories)))

    variables.append("answerPointer")
    parents["answerPointer"] = [
        "schemaName",
        *[f"Object.{cat_id}.Ordinal.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.QueryOrdinal" for cat_id in range(num_categories)],
        "answerCategory",
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("answer")
    parents["answer"] = [
        "schemaName",
        "answerPointer",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["answer"] = []
    for cat_id in range(1, num_categories):
        values["answer"].extend(first_schema.items[first_schema.categories[cat_id]])
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    return CausalModel(variables, values, parents, mechanisms)


def multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
    schemas: List[Schema],
    num_instances: int,
    order: list[int] | None = None,
    fillers: bool = False,
    num_fillers_per_item: int = 1,
):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schemas[0].categories)

    if order is None:
        order = list(range(num_categories))

    assert all(
        len(schema.categories) == num_categories for schema in schemas
    ), "All schemas must have the same number of categories"
    assert len(order) == num_categories and len(order) == len(
        set(order)
    ), "Order must be the same length as the number of categories and must be a permutation"

    schema_by_name = {schema.name: schema for schema in schemas}

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(schema_name, *args):
        """
        - schemaName
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        schema = schema_by_name[schema_name]

        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_answer(schema_name, answer_pointer, answer_category, *final_objects):
        """
        - schemaName
        - answerPointer
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(final_objects[index])

        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            return "unknown"

        answer_category_id = schema.categories.index(answer_category)
        return objects[answer_category_id][answer_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(schema_name, answer_category, *args):
        """
        - schemaName
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(
            schema,
            data,
            f"ordering_{''.join(str(i) for i in order)}",
            fillers=fillers,
            num_fillers_per_item=num_fillers_per_item,
        )

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]
        query_index = objects[used_cats_in_queries[0]].index(first_query)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\n{data}"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}

    first_schema = schemas[0]
    schema_categories = [first_schema.categories[i] for i in order]
    for cat_id in range(num_categories):
        # We currently use the first schema's values for the values and mechanisms dict since it only matters when
        # creating datasets, in which case we'll only have one schema.
        cur_cat = schema_categories[cat_id]
        for i in range(num_instances):

            # Add the base variables
            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = first_schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(first_schema.items[cur_cat])

            # Add the ordinal variables
            variables.append(f"Object.{cat_id}.Ordinal.{i}")
            values[f"Object.{cat_id}.Ordinal.{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cat_id == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"Object.{cat_id}.Ordinal.{i}"] = []
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"Object.{cat_id}.Ordinal.{i}"] = [f"Object.{x}.Ordinal.{i}" for x in range(cat_id)]
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = identity

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = first_schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(first_schema.items[cur_cat])

        variables.append(f"Object.{cat_id}.QueryOrdinal")
        parents[f"Object.{cat_id}.QueryOrdinal"] = [
            *[f"Object.{cat_id}.{i}" for i in range(num_instances)],
            f"Object.{cat_id}.Query",
        ]
        values[f"Object.{cat_id}.QueryOrdinal"] = list(range(num_instances))
        mechanisms[f"Object.{cat_id}.QueryOrdinal"] = get_cat_query_ordinal

    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [schema.name for schema in schemas]
    mechanisms["schemaName"] = lambda: random.choice([schema.name for schema in schemas])

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = list(range(num_categories))
    mechanisms["answerCategory"] = lambda: random.choice(list(range(num_categories)))

    variables.append("answerPointer")
    parents["answerPointer"] = [
        "schemaName",
        *[f"Object.{cat_id}.Ordinal.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.QueryOrdinal" for cat_id in range(num_categories)],
        "answerCategory",
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("answer")
    parents["answer"] = [
        "schemaName",
        "answerPointer",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["answer"] = []
    for cat_id in range(1, num_categories):
        values["answer"].extend(first_schema.items[schema_categories[cat_id]])
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    return CausalModel(variables, values, parents, mechanisms)


def multi_order_multi_schema_task_to_lookbacks_generic_causal_model_with_special_vars(
    schemas: List[Schema], num_instances: int, order: list[int] | None = None
):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schemas[0].categories)

    if order is None:
        order = list(range(num_categories))

    assert all(
        len(schema.categories) == num_categories for schema in schemas
    ), "All schemas must have the same number of categories"
    assert len(order) == num_categories and len(order) == len(
        set(order)
    ), "Order must be the same length as the number of categories and must be a permutation"

    schema_by_name = {schema.name: schema for schema in schemas}

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(schema_name, *args):
        """
        - schemaName
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        schema = schema_by_name[schema_name]

        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_answer(schema_name, answer_pointer, answer_category, *final_objects):
        """
        - schemaName
        - answerPointer
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(final_objects[index])

        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            return "unknown"

        answer_category_id = schema.categories.index(answer_category)
        return objects[answer_category_id][answer_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(schema_name, answer_category, *args):
        """
        - schemaName
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, f"ordering_{''.join(str(i) for i in order)}")

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]
        query_index = objects[used_cats_in_queries[0]].index(first_query)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\n{data}"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}

    first_schema = schemas[0]
    schema_categories = [first_schema.categories[i] for i in order]
    for cat_id in range(num_categories):
        # We currently use the first schema's values for the values and mechanisms dict since it only matters when
        # creating datasets, in which case we'll only have one schema.
        cur_cat = schema_categories[cat_id]
        for i in range(num_instances):

            # Add the base variables
            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = first_schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(first_schema.items[cur_cat])

            # Add the ordinal variables
            variables.append(f"Object.{cat_id}.Ordinal.{i}")
            values[f"Object.{cat_id}.Ordinal.{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cat_id == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"Object.{cat_id}.Ordinal.{i}"] = []
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"Object.{cat_id}.Ordinal.{i}"] = [f"Object.{x}.Ordinal.{i}" for x in range(cat_id)]
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = identity

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = first_schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(first_schema.items[cur_cat])

        variables.append(f"Object.{cat_id}.QueryOrdinal")
        parents[f"Object.{cat_id}.QueryOrdinal"] = [
            *[f"Object.{cat_id}.{i}" for i in range(num_instances)],
            f"Object.{cat_id}.Query",
        ]
        values[f"Object.{cat_id}.QueryOrdinal"] = list(range(num_instances))
        mechanisms[f"Object.{cat_id}.QueryOrdinal"] = get_cat_query_ordinal

    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [schema.name for schema in schemas]
    mechanisms["schemaName"] = lambda: random.choice([schema.name for schema in schemas])

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = list(range(num_categories))
    mechanisms["answerCategory"] = lambda: random.choice(list(range(num_categories)))

    variables.append("answerPointer")
    parents["answerPointer"] = [
        "schemaName",
        *[f"Object.{cat_id}.Ordinal.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.QueryOrdinal" for cat_id in range(num_categories)],
        "answerCategory",
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("answer")
    parents["answer"] = [
        "schemaName",
        "answerPointer",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["answer"] = []
    for cat_id in range(1, num_categories):
        values["answer"].extend(first_schema.items[schema_categories[cat_id]])
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    ### <shenanigans> ###

    variables.append("isFirstTwo")
    variables.append("isLast")

    parents["isFirstTwo"] = ["answerPointer"]
    values["isFirstTwo"] = [True, False]
    mechanisms["isFirstTwo"] = lambda answerPointer: answerPointer == 0 or answerPointer == 1

    parents["isLast"] = ["answerPointer"]
    values["isLast"] = [True, False]
    mechanisms["isLast"] = lambda answerPointer: answerPointer == num_instances - 1

    ### </shenanigans> ###

    return CausalModel(variables, values, parents, mechanisms)


def multi_order_multi_schema_task_to_lookbacks_keyload_causal_model(
    schemas: List[Schema], num_instances: int, order: list[int] | None = None
):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schemas[0].categories)

    if order is None:
        order = list(range(num_categories))

    assert all(
        len(schema.categories) == num_categories for schema in schemas
    ), "All schemas must have the same number of categories"
    assert len(order) == num_categories and len(order) == len(
        set(order)
    ), "Order must be the same length as the number of categories and must be a permutation"

    schema_by_name = {schema.name: schema for schema in schemas}

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(schema_name, *args):
        """
        - schemaName
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        schema = schema_by_name[schema_name]

        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_key(schema_name, answer_category, *args):
        """
        Returns the value of the first key category that is not the answer category.
        - schemaName
        - answerCategory
        - Object.i.0, ..., Object.i.N
        """
        objects = args
        schema = schema_by_name[schema_name]
        answer_category_id = schema.categories.index(answer_category)

        key_categories = [i for i in range(num_categories) if i != answer_category_id]

        first_key_category = key_categories[0]
        return objects[first_key_category]

    def get_answer(schema_name, query_key, answer_category, *args):
        """
        - QueryKey
        - answerCategory
        - Key.0,Key.1,...,Key.N
        - Object.answerCategory.0, Object.answerCategory.1, ..., Object.answerCategory.N
        """
        keys = args[:num_instances]
        schema = schema_by_name[schema_name]
        answer_category_id = schema.categories.index(answer_category)

        object_values = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                object_values[cat_id].append(args[num_instances + index])

        for i in range(num_instances):
            if keys[i] == query_key:
                return object_values[answer_category_id][i]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(schema_name, answer_category, *args):
        """
        - schemaName
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, f"ordering_{''.join(str(i) for i in order)}")

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]

        query_index = objects[used_cats_in_queries[0]].index(first_query)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\n{data}"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}

    first_schema = schemas[0]
    schema_categories = [first_schema.categories[i] for i in order]
    for cat_id in range(num_categories):
        # We currently use the first schema's values for the values and mechanisms dict since it only matters when
        # creating datasets, in which case we'll only have one schema.
        cur_cat = schema_categories[cat_id]
        for i in range(num_instances):

            # Add the base variables
            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = first_schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(first_schema.items[cur_cat])

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = first_schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(first_schema.items[cur_cat])

    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [schema.name for schema in schemas]
    mechanisms["schemaName"] = lambda: random.choice([schema.name for schema in schemas])

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = list(range(num_categories))
    mechanisms["answerCategory"] = lambda: random.choice(list(range(num_categories)))

    for i in range(num_instances):
        variables.append(f"Key.{i}")
        parents[f"Key.{i}"] = [
            "schemaName",
            "answerCategory",
            *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories)],
        ]
        values[f"Key.{i}"] = []
        mechanisms[f"Key.{i}"] = get_key

    variables.append("QueryKey")
    parents["QueryKey"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["QueryKey"] = []
    mechanisms["QueryKey"] = get_key

    variables.append("answer")
    parents["answer"] = [
        "schemaName",
        "QueryKey",
        "answerCategory",
        *[f"Key.{i}" for i in range(num_instances)],
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["answer"] = []
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    return CausalModel(variables, values, parents, mechanisms)


def multi_order_multi_schema_task_to_lookbacks_generic_causal_model_with_pdfs(
    schemas: List[Schema], num_instances: int, order: list[int] | None = None, pdfs: list[float] | None = None
):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schemas[0].categories)

    if order is None:
        order = list(range(num_categories))

    assert all(
        len(schema.categories) == num_categories for schema in schemas
    ), "All schemas must have the same number of categories"
    assert len(order) == num_categories and len(order) == len(
        set(order)
    ), "Order must be the same length as the number of categories and must be a permutation"

    schema_by_name = {schema.name: schema for schema in schemas}

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(schema_name, *args):
        """
        - schemaName
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        schema = schema_by_name[schema_name]

        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_answer(schema_name, answer_pointer, answer_category, *final_objects):
        """
        - schemaName
        - answerPointer
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(final_objects[index])

        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            return "unknown"

        answer_category_id = schema.categories.index(answer_category)
        return objects[answer_category_id][answer_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(schema_name, answer_category, *args):
        """
        - schemaName
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, f"ordering_{''.join(str(i) for i in order)}")

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]
        query_index = objects[used_cats_in_queries[0]].index(first_query)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\n{data}"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    variables = []
    parents = {}
    values = {}
    mechanisms = {}

    first_schema = schemas[0]
    schema_categories = [first_schema.categories[i] for i in order]
    for cat_id in range(num_categories):
        # We currently use the first schema's values for the values and mechanisms dict since it only matters when
        # creating datasets, in which case we'll only have one schema.
        cur_cat = schema_categories[cat_id]
        for i in range(num_instances):

            # Add the base variables
            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = first_schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(first_schema.items[cur_cat])

            # Add the ordinal variables
            variables.append(f"Object.{cat_id}.Ordinal.{i}")
            values[f"Object.{cat_id}.Ordinal.{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cat_id == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"Object.{cat_id}.Ordinal.{i}"] = []
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"Object.{cat_id}.Ordinal.{i}"] = [f"Object.{x}.Ordinal.{i}" for x in range(cat_id)]
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = identity

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = first_schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(first_schema.items[cur_cat])

        variables.append(f"Object.{cat_id}.QueryOrdinal")
        parents[f"Object.{cat_id}.QueryOrdinal"] = [
            *[f"Object.{cat_id}.{i}" for i in range(num_instances)],
            f"Object.{cat_id}.Query",
        ]
        values[f"Object.{cat_id}.QueryOrdinal"] = list(range(num_instances))
        mechanisms[f"Object.{cat_id}.QueryOrdinal"] = get_cat_query_ordinal

    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [schema.name for schema in schemas]
    mechanisms["schemaName"] = lambda: random.choice([schema.name for schema in schemas])

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = list(range(num_categories))
    mechanisms["answerCategory"] = lambda: random.choice(list(range(num_categories)))

    variables.append("answerPointer")
    parents["answerPointer"] = [
        "schemaName",
        *[f"Object.{cat_id}.Ordinal.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.QueryOrdinal" for cat_id in range(num_categories)],
        "answerCategory",
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("answer")
    parents["answer"] = [
        "schemaName",
        "answerPointer",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["answer"] = []
    for cat_id in range(1, num_categories):
        values["answer"].extend(first_schema.items[schema_categories[cat_id]])
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda *args: " ".join(args)

    return CausalModel(variables, values, parents, mechanisms)


def multi_order_multi_schema_task_to_lookbacks_complete_causal_model(
    schemas: List[Schema],
    num_instances: int,
    order: list[int] | None = None,
    fillers: bool = False,
    do_sample: bool = True,
    return_probs: bool = False,
    atticus_baseline=False,
):
    """
    Convert a task to a causal model.
    """
    # Note that when using the same model across schemas they have to have the same number of categories
    num_categories = len(schemas[0].categories)

    if order is None:
        order = list(range(num_categories))

    assert all(
        len(schema.categories) == num_categories for schema in schemas
    ), "All schemas must have the same number of categories"
    assert len(order) == num_categories and len(order) == len(
        set(order)
    ), "Order must be the same length as the number of categories and must be a permutation"

    schema_by_name = {schema.name: schema for schema in schemas}

    learned_model = get_learned_model(num_instances)

    def get_cat_query_ordinal(*args):
        """
        Args:
        - [all_object1, all_object2, ...]
        - object query
        """
        all_objects = args[:num_instances]
        object_query = args[-1]

        # Find matching agent and return corresponding state ordinal
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i

        return None

    def get_answer_pointer(schema_name, *args):
        """
        - schemaName
        - Object.0.Ordinal.0,Object.0.Ordinal.1,...,Object.N.Ordinal.0,Object.N.Ordinal.1,..Object.N.Ordinal.M
        - Object.0.QueryOrdinal,Object.1.QueryOrdinal,...,Object.N.QueryOrdinal
        - answerCategory
        """
        schema = schema_by_name[schema_name]

        ordinals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                ordinals[cat_id].append(args[index])

        query_ordinals = args[num_categories * num_instances : -1]
        used_query_ordinals = [qo for qo in query_ordinals if qo is not None]
        answer_category = args[-1]
        answer_category_id = schema.categories.index(answer_category)

        # If query ordinals match, return the state ordinal at that index
        if all(q == used_query_ordinals[0] for q in used_query_ordinals):
            return ordinals[answer_category_id][used_query_ordinals[0]]

        return "unknown"

    def get_answer(schema_name, answer_pointer, lexical_pointer, answer_category, *final_objects):
        """
        - schemaName
        - answerPointer
        - lexicalPointer
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(final_objects[index])

        # Return the state at the index specified by answer_pointer
        if answer_pointer == "unknown":
            assert False
            return "unknown"

        answer_category_id = schema.categories.index(answer_category)

        if atticus_baseline:
            if answer_pointer in [0, 1, 2, 3, num_instances - 1]:
                return objects[answer_category_id][answer_pointer]
            else:
                return objects[answer_category_id][lexical_pointer]

        logits = learned_model(torch.tensor([[answer_pointer, lexical_pointer]]))[0]

        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            return {objects[answer_category_id][i]: probs[i].item() for i in range(num_instances)}

        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            pred_pointer = torch.multinomial(probs, num_samples=1).item()
        else:
            pred_pointer = logits.argmax().item()

        return objects[answer_category_id][pred_pointer]

    # def raw_input(agent0, agent1, state0, state1, object0, object1, agent_query, object_query):
    def raw_input(schema_name, answer_category, *args):
        """
        - schemaName
        - answerCategory
        - Object.0.0,Object.0.1,...,Object.N.0,...Object.N.M
        - Object.0.Query,Object.1.Query
        """
        schema = schema_by_name[schema_name]

        objects = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                objects[cat_id].append(args[index])

        queries = args[num_categories * num_instances :]

        data = list(zip(*list(objects.values())))

        if schema.templates.prefix:
            all_items_formatted = {
                schema.categories[cat] + "_list": _format_list(items) for cat, items in objects.items()
            }
            prefix = schema.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        context = define_by_key(schema, data, f"ordering_{''.join(str(i) for i in order)}", fillers=fillers)

        used_cats_in_queries = [i for i, q in enumerate(queries) if q]
        cats_to_queries = {i: q for i, q in enumerate(queries) if q}

        # Get query index and assert that all query objects are in the same position
        first_query = cats_to_queries[used_cats_in_queries[0]]
        query_index = objects[used_cats_in_queries[0]].index(first_query)
        for cat in used_cats_in_queries:
            cur_query = cats_to_queries[cat]
            assert (
                objects[cat].index(cur_query) == query_index
            ), f"Query objects must allude to same position: {cur_query} and {first_query} [{prefix}{context}]\n\n{data}"

        # Get the query mapping by sorted list of categories used to query
        sorted_query_categories = sorted([schema.categories[i] for i in used_cats_in_queries])

        answer_category_id = schema.categories.index(answer_category)
        selected_query = schema.templates.queries[
            f"Q:{'_'.join(sorted_query_categories)} A:{schema.categories[answer_category_id]}"
        ]

        query_mapping = _get_instance_mapping(query_index, data, schema.categories)
        question = selected_query.question.format(**query_mapping)

        return f"{prefix}{context} {question} Answer:"

    def constant(i):
        return i

    # def get_lexical_query(schema_name, answer_category, *queries):
    #     """
    #     - schemaName
    #     - answerCategory
    #     - Object.0.Query,Object.1.Query,...,Object.N.Query
    #     """
    #     return queries

    def get_lexical_pointer(lexical_query, *args):
        """
        - Object.0.Lexical.0,Object.0.Lexical.1,...,Object.N.Lexical.0,Object.N.Lexical.1,..Object.N.Lexical.M
        - Object.0.Query,Object.1.Query,...,Object.N.Query
        """
        lexicals = DefaultDict(list)
        for cat_id in range(num_categories):
            for instance in range(num_instances):
                index = cat_id * num_instances + instance
                lexicals[cat_id].append(args[index])

        for cat_id in range(num_categories):
            for instance in range(num_instances):
                for lexical_cat_id in range(num_categories):
                    if lexical_query[cat_id] in lexicals[lexical_cat_id][instance]:
                        return instance
        return "unknown"

    variables = []
    parents = {}
    values = {}
    mechanisms = {}

    first_schema = schemas[0]
    schema_categories = [first_schema.categories[i] for i in order]
    for cat_id in range(num_categories):
        # We currently use the first schema's values for the values and mechanisms dict since it only matters when
        # creating datasets, in which case we'll only have one schema.
        cur_cat = schema_categories[cat_id]
        for i in range(num_instances):

            # Add the base variables
            variables.append(f"Object.{cat_id}.{i}")
            parents[f"Object.{cat_id}.{i}"] = []
            values[f"Object.{cat_id}.{i}"] = first_schema.items[cur_cat]
            mechanisms[f"Object.{cat_id}.{i}"] = lambda: random.choice(first_schema.items[cur_cat])

            # Add lexical variables
            variables.append(f"Object.{cat_id}.Lexical.{i}")
            # The lexical variables have the previous variables as parents
            parents[f"Object.{cat_id}.Lexical.{i}"] = [f"Object.{j}.{i}" for j in range(cat_id)]
            values[f"Object.{cat_id}.Lexical.{i}"] = []
            mechanisms[f"Object.{cat_id}.Lexical.{i}"] = identity_list

            # Add the ordinal variables
            variables.append(f"Object.{cat_id}.Ordinal.{i}")
            values[f"Object.{cat_id}.Ordinal.{i}"] = list(range(num_instances))

            # Our current causal model assumes a copying fromt the first object to the rest of the OI
            if cat_id == 0:
                # Therefore the first ordinal variable has no parents
                parents[f"Object.{cat_id}.Ordinal.{i}"] = []
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = partial(constant, i)
            else:
                # The rest of the ordinal variables have the first ordinal variable as a parent
                parents[f"Object.{cat_id}.Ordinal.{i}"] = [f"Object.{x}.Ordinal.{i}" for x in range(cat_id)]
                mechanisms[f"Object.{cat_id}.Ordinal.{i}"] = identity

        # The query variables have no dependencies
        variables.append(f"Object.{cat_id}.Query")
        parents[f"Object.{cat_id}.Query"] = []
        values[f"Object.{cat_id}.Query"] = first_schema.items[cur_cat]
        mechanisms[f"Object.{cat_id}.Query"] = lambda: random.choice(first_schema.items[cur_cat])

        variables.append(f"Object.{cat_id}.QueryOrdinal")
        parents[f"Object.{cat_id}.QueryOrdinal"] = [
            *[f"Object.{cat_id}.{i}" for i in range(num_instances)],
            f"Object.{cat_id}.Query",
        ]
        values[f"Object.{cat_id}.QueryOrdinal"] = list(range(num_instances))
        mechanisms[f"Object.{cat_id}.QueryOrdinal"] = get_cat_query_ordinal

    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [schema.name for schema in schemas]
    mechanisms["schemaName"] = lambda: random.choice([schema.name for schema in schemas])

    variables.append("answerCategory")
    parents["answerCategory"] = []
    values["answerCategory"] = []
    mechanisms["answerCategory"] = []

    variables.append("answerPointer")
    parents["answerPointer"] = [
        "schemaName",
        *[f"Object.{cat_id}.Ordinal.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.QueryOrdinal" for cat_id in range(num_categories)],
        "answerCategory",
    ]
    values["answerPointer"] = list(range(num_instances))
    mechanisms["answerPointer"] = get_answer_pointer

    variables.append("lexicalQuery")
    parents["lexicalQuery"] = [
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["lexicalQuery"] = []
    mechanisms["lexicalQuery"] = identity_list

    variables.append("lexicalPointer")
    parents["lexicalPointer"] = [
        "lexicalQuery",
        *[f"Object.{cat_id}.Lexical.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["lexicalPointer"] = list(range(num_instances))
    mechanisms["lexicalPointer"] = get_lexical_pointer

    variables.append("answer")
    parents["answer"] = [
        "schemaName",
        "answerPointer",
        "lexicalPointer",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
    ]
    values["answer"] = []
    for cat_id in range(1, num_categories):
        values["answer"].extend(first_schema.items[schema_categories[cat_id]])
    mechanisms["answer"] = get_answer

    variables.append("raw_input")
    parents["raw_input"] = [
        "schemaName",
        "answerCategory",
        *[f"Object.{cat_id}.{i}" for cat_id in range(num_categories) for i in range(num_instances)],
        *[f"Object.{cat_id}.Query" for cat_id in range(num_categories)],
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = raw_input

    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = (lambda *args: " ".join(args)) if not return_probs else (lambda *args: args)

    return CausalModel(variables, values, parents, mechanisms)


from functools import partial
from collections import defaultdict as DefaultDict
import random
from typing import List

INACTIVE = "__inactive__"


def multi_order_multi_schema_task_to_lookbacks_generic_causal_model_mixed(
    schemas: List[Schema],
    num_instances: int,
    orders: dict[str, list[int]] | None = None,
    fillers: bool = False,
    num_fillers_per_item: int = 1,
):
    """
    Generic-variable version that supports schemas with different #categories.

    - Uses fixed generic variable names up to K_max = max(len(schema.categories)).
    - Per-sample, a single schema is chosen -> its categories are packed into slots [0..C-1]
      following the provided order (or natural order). Slots [C..K_max-1] are inactive.
    - New per-sample helper variables:
        * active.{cat}: bool mask
        * categoryName.{cat}: category name or "__inactive__"
        * numActive: number of active categories (C)
    """

    # ---------------------------
    # Global bounds and indexing
    # ---------------------------
    K_max = max(len(s.categories) for s in schemas)
    schema_by_name = {s.name: s for s in schemas}

    # ---------------------------
    # Helpers
    # ---------------------------
    def _constant(i):
        return i

    def _schema_picker(schema_names):
        return random.choice(schema_names)

    def _num_active(schema_name):
        return len(schema_by_name[schema_name].categories)

    def _category_name(schema_name, cat_id, orders):
        sch = schema_by_name[schema_name]
        C = len(sch.categories)
        if cat_id >= C:
            return INACTIVE
        order = (orders or {}).get(schema_name) or list(range(C))
        # validate permutation
        assert sorted(order) == list(range(C)), f"orders[{schema_name}] must be a permutation of 0..{C-1}"
        return sch.categories[order[cat_id]]

    def _is_active(schema_name, cat_id):
        sch = schema_by_name[schema_name]
        return cat_id < len(sch.categories)

    def _rand_item(schema_name, cat_id, orders):
        name = _category_name(schema_name, cat_id, orders)
        if name == INACTIVE:
            return None
        sch = schema_by_name[schema_name]
        return random.choice(sch.items[name])

    def _get_cat_query_ordinal(num_instances, *args):
        """
        Args:
          - [all_object_i for i in 0..num_instances-1]
          - object_query (last)
        """
        all_objects = args[:num_instances]
        object_query = args[-1]
        for i, obj in enumerate(all_objects):
            if obj == object_query:
                return i
        return None

    def _get_answer_pointer_generic(*args, num_instances=num_instances, K_max=K_max):
        """
        Parents (in this exact order):
        - numActive
        - categoryName.0 .. categoryName.(K_max-1)
        - Object.{cat}.Ordinal.{i}  for cat in 0..K_max-1, i in 0..num_instances-1
        - Object.{cat}.QueryOrdinal  for cat in 0..K_max-1
        - answerCategory (NAME)
        - schemaName   <-- last
        """
        # unpack
        idx = 0
        num_active = args[idx]
        idx += 1
        cat_names = list(args[idx : idx + K_max])
        idx += K_max

        ordinals = DefaultDict(list)
        for cat_id in range(K_max):
            for _ in range(num_instances):
                ordinals[cat_id].append(args[idx])
                idx += 1

        query_ordinals = list(args[idx : idx + K_max])
        idx += K_max
        answer_category = args[idx]
        idx += 1
        schema_name = args[idx]  # last

        sch = schema_by_name[schema_name]
        C = min(num_active, len(sch.categories))

        # map answer category name -> packed cat_id
        try:
            answer_cat_id = cat_names.index(answer_category)
        except ValueError:
            return "unknown"

        used_query_ordinals = [qo for qo in query_ordinals[:C] if qo is not None]
        if not used_query_ordinals:
            return "unknown"
        if any(q != used_query_ordinals[0] for q in used_query_ordinals):
            return "unknown"

        qi = used_query_ordinals[0]
        if qi is None or not (0 <= qi < num_instances):
            return "unknown"
        return ordinals[answer_cat_id][qi]

    def _get_answer_generic(*args, num_instances=num_instances, K_max=K_max):
        """
        Parents:
        - answerPointer
        - answerCategory
        - Object.{cat}.{i} (K_max * num_instances)
        - numActive
        - categoryName.0 .. categoryName.(K_max-1)
        - schemaName  <-- last
        """
        idx = 0
        answer_pointer = args[idx]
        idx += 1
        answer_category = args[idx]
        idx += 1

        objs_flat = list(args[idx : idx + K_max * num_instances])
        idx += K_max * num_instances
        num_active = args[idx]
        idx += 1
        cat_names = list(args[idx : idx + K_max])
        idx += K_max
        schema_name = args[idx]  # last

        if answer_pointer == "unknown":
            return "unknown"

        # rebuild per-cat objects
        objects = DefaultDict(list)
        k = 0
        for cat_id in range(K_max):
            for _ in range(num_instances):
                objects[cat_id].append(objs_flat[k])
                k += 1

        try:
            answer_cat_id = cat_names.index(answer_category)
        except ValueError:
            return "unknown"

        if answer_cat_id >= num_active:
            return "unknown"
        return objects[answer_cat_id][answer_pointer]

    def _raw_input_generic(*args, order_lookup=orders, num_instances=num_instances, K_max=K_max):
        """
        Parents:
        - answerCategory
        - Object.{cat}.{i} (K_max * num_instances)
        - Object.{cat}.Query (K_max)
        - numActive
        - categoryName.0 .. categoryName.(K_max-1)
        - schemaName  <-- last
        """
        idx = 0
        answer_category = args[idx]
        idx += 1
        objs_flat = list(args[idx : idx + K_max * num_instances])
        idx += K_max * num_instances
        queries = list(args[idx : idx + K_max])
        idx += K_max
        num_active = args[idx]
        idx += 1
        cat_names = list(args[idx : idx + K_max])
        idx += K_max
        schema_name = args[idx]  # last

        sch = schema_by_name[schema_name]
        C = min(num_active, len(sch.categories))
        if C == 0:
            return "Answer:"

        # rebuild objects per cat
        objects = DefaultDict(list)
        k = 0
        for cat_id in range(K_max):
            for _ in range(num_instances):
                objects[cat_id].append(objs_flat[k])
                k += 1

        active_objects = {c: objects[c] for c in range(C)}
        active_queries = queries[:C]
        active_cat_names = cat_names[:C]

        data = list(zip(*[active_objects[c] for c in range(C)]))

        if sch.templates.prefix:
            all_items_formatted = {
                active_cat_names[cat] + "_list": _format_list(items) for cat, items in active_objects.items()
            }
            prefix = sch.templates.prefix.format(**all_items_formatted)
        else:
            prefix = ""

        order = (order_lookup or {}).get(schema_name) or list(range(C))
        context = define_by_key(
            sch,
            data,
            f"ordering_{''.join(str(i) for i in order)}",
            fillers=fillers,
            num_fillers_per_item=num_fillers_per_item,
        )

        used_cats = [i for i, q in enumerate(active_queries) if q]
        if not used_cats:
            return f"{prefix}{context} Answer:"

        first_q = active_queries[used_cats[0]]
        query_index = active_objects[used_cats[0]].index(first_q)
        for cat in used_cats:
            cur_q = active_queries[cat]
            assert active_objects[cat].index(cur_q) == query_index, (
                f"Query objects must allude to same position: {cur_q} and {first_q} " f"[{prefix}{context}]\n\n{data}"
            )

        sorted_query_categories = sorted([active_cat_names[i] for i in used_cats])
        selected_query = sch.templates.queries[f"Q:{'_'.join(sorted_query_categories)} A:{answer_category}"]
        query_mapping = _get_instance_mapping(query_index, data, active_cat_names)
        question = selected_query.question.format(**query_mapping)
        return f"{prefix}{context} {question} Answer:"

    # ---------------------------
    # Graph assembly
    # ---------------------------
    variables, parents, values, mechanisms = [], {}, {}, {}

    # schemaName
    variables.append("schemaName")
    parents["schemaName"] = []
    values["schemaName"] = [s.name for s in schemas]
    mechanisms["schemaName"] = partial(_schema_picker, values["schemaName"])

    # numActive
    variables.append("numActive")
    parents["numActive"] = ["schemaName"]
    values["numActive"] = list(range(K_max + 1))
    mechanisms["numActive"] = _num_active

    # categoryName.{cat}, active.{cat}
    for cat in range(K_max):
        v_cname = f"categoryName.{cat}"
        variables.append(v_cname)
        parents[v_cname] = ["schemaName"]
        values[v_cname] = None  # dynamic names per schema
        mechanisms[v_cname] = partial(_category_name, cat_id=cat, orders=orders)

        v_active = f"active.{cat}"
        variables.append(v_active)
        parents[v_active] = ["schemaName"]
        values[v_active] = [True, False]
        mechanisms[v_active] = partial(_is_active, cat_id=cat)

    # Objects + Ordinals + Queries (generic)
    for cat in range(K_max):
        # Object.{cat}.{i}
        for i in range(num_instances):
            v = f"Object.{cat}.{i}"
            variables.append(v)
            parents[v] = ["schemaName"]
            values[v] = None  # dynamic, depends on schema & category
            mechanisms[v] = partial(_rand_item, cat_id=cat, orders=orders)

        # Object.{cat}.Ordinal.{i}
        for i in range(num_instances):
            v = f"Object.{cat}.Ordinal.{i}"
            variables.append(v)
            values[v] = list(range(num_instances))
            if cat == 0:
                parents[v] = []
                mechanisms[v] = partial(_constant, i)
            else:
                parents[v] = [f"Object.{x}.Ordinal.{i}" for x in range(cat)]
                mechanisms[v] = identity

        # Object.{cat}.Query
        vq = f"Object.{cat}.Query"
        variables.append(vq)
        parents[vq] = ["schemaName"]
        values[vq] = None
        mechanisms[vq] = partial(_rand_item, cat_id=cat, orders=orders)

        # Object.{cat}.QueryOrdinal
        vqo = f"Object.{cat}.QueryOrdinal"
        variables.append(vqo)
        parents[vqo] = [
            *[f"Object.{cat}.{i}" for i in range(num_instances)],
            f"Object.{cat}.Query",
        ]
        values[vqo] = list(range(num_instances))
        mechanisms[vqo] = partial(_get_cat_query_ordinal, num_instances)

    # answerCategory (NAME among active categories)
    def _answer_category(schema_name):
        sch = schema_by_name[schema_name]
        return random.choice(list(sch.categories))

    variables.append("answerCategory")
    parents["answerCategory"] = ["schemaName"]
    values["answerCategory"] = None
    mechanisms["answerCategory"] = _answer_category

    # answerPointer
    variables.append("answerPointer")
    parents["answerPointer"] = [
        "numActive",
        *[f"categoryName.{c}" for c in range(K_max)],
        *[f"Object.{cat}.Ordinal.{i}" for cat in range(K_max) for i in range(num_instances)],
        *[f"Object.{cat}.QueryOrdinal" for cat in range(K_max)],
        "answerCategory",
        "schemaName",
    ]
    values["answerPointer"] = list(range(num_instances)) + ["unknown"]
    # mechanisms["answerPointer"] = partial(_get_answer_pointer_generic, num_instances=num_instances)
    mechanisms["answerPointer"] = _get_answer_pointer_generic

    # answer
    variables.append("answer")
    parents["answer"] = [
        "answerPointer",
        "answerCategory",
        *[f"Object.{cat}.{i}" for cat in range(K_max) for i in range(num_instances)],
        "numActive",
        *[f"categoryName.{c}" for c in range(K_max)],
        "schemaName",
    ]
    values["answer"] = None
    mechanisms["answer"] = _get_answer_generic

    # raw_input
    variables.append("raw_input")
    parents["raw_input"] = [
        "answerCategory",
        *[f"Object.{cat}.{i}" for cat in range(K_max) for i in range(num_instances)],
        *[f"Object.{cat}.Query" for cat in range(K_max)],
        "numActive",
        *[f"categoryName.{c}" for c in range(K_max)],
        "schemaName",
    ]
    values["raw_input"] = None
    mechanisms["raw_input"] = _raw_input_generic

    # raw_output
    variables.append("raw_output")
    parents["raw_output"] = ["answer"]
    values["raw_output"] = None
    mechanisms["raw_output"] = lambda a: a if a is not None else "unknown"

    return CausalModel(variables, values, parents, mechanisms)
