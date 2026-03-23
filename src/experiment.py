"""
Manipulation of Hard-to-Describe Directions
============================================
Core experiment: Tests whether learned token embeddings can manipulate residual
stream directions that resist explicit prompting, and whether the advantage of
learned tokens increases as features become harder to describe.

Model: Pythia-410M (via TransformerLens)
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path("/workspaces/hard-to-describe-dir-claude")
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Feature Definitions ──────────────────────────────────────────────────────
# Each feature has contrastive text pairs and multiple prompting attempts.
# Features are ordered from easy-to-describe to hard-to-describe.

FEATURES = {
    "sentiment": {
        "description": "Positive vs negative emotional tone",
        "expected_difficulty": "easy",
        "positive_examples": [
            "I absolutely love this beautiful sunny day! Everything is going wonderfully.",
            "This is the best experience I've ever had. I'm so happy and grateful.",
            "What a fantastic movie! The acting was superb and the story was heartwarming.",
            "I'm thrilled about the promotion. My hard work finally paid off!",
            "The garden is blooming magnificently. Nature is truly wonderful.",
            "Our team won the championship! I'm bursting with joy and pride.",
            "The food was absolutely delicious, every bite was heavenly.",
            "I feel so lucky to have such amazing friends in my life.",
            "This vacation has been perfect in every way. Paradise on earth!",
            "The concert was incredible, an unforgettable night of pure magic.",
            "My new puppy brings so much happiness to our household.",
            "I got into my dream school! All those late nights studying were worth it.",
            "The sunset tonight was breathtakingly beautiful, painting the sky gold.",
            "I'm grateful for this wonderful community and all the support.",
            "Today was one of the happiest days of my life, full of laughter.",
            "The book was captivating from start to finish, a true masterpiece.",
            "I'm so proud of how far we've come together as a team.",
            "The weather is perfect and I feel absolutely energized and alive.",
            "Such a wonderful surprise! This made my entire week better.",
            "I love spending time with family. These moments are priceless.",
            "The presentation went brilliantly and everyone applauded.",
            "I feel on top of the world today, everything is clicking.",
            "What a delightful conversation, I learned so much and had fun.",
            "The flowers smell divine, filling the room with sweet fragrance.",
            "I'm excited about the future, so many wonderful possibilities ahead.",
        ],
        "negative_examples": [
            "This is the worst day ever. Everything keeps going wrong.",
            "I'm devastated by the terrible news. I can't stop crying.",
            "What an awful movie. Complete waste of time and money.",
            "I'm furious about the unfair treatment. This is outrageous.",
            "The weather is miserable and I feel completely drained.",
            "Our team lost badly. I'm crushed and disappointed.",
            "The food was disgusting, completely inedible and overpriced.",
            "I feel so alone and abandoned by everyone I trusted.",
            "This trip has been a nightmare from start to finish.",
            "The concert was terrible, the worst performance I've ever seen.",
            "My old car broke down again. I'm so frustrated and angry.",
            "I got rejected from every school I applied to. I'm heartbroken.",
            "The storm destroyed everything, it's a devastating sight.",
            "I'm sick of this toxic environment and all the negativity.",
            "Today was one of the worst days, nothing but problems.",
            "The book was boring, poorly written garbage not worth reading.",
            "I'm ashamed of how badly our project failed this quarter.",
            "The weather is awful and I feel miserable and hopeless.",
            "Such terrible news. This ruined my entire week completely.",
            "I hate being stuck inside with nothing to do all day.",
            "The presentation was a disaster and everyone was disappointed.",
            "I feel like everything is falling apart around me.",
            "What a depressing conversation, nothing but bad news and complaints.",
            "The smell in here is revolting, making me feel sick.",
            "I dread the future, everything looks bleak and hopeless.",
        ],
        "prompts": [
            "Write in a very positive and happy tone:",
            "Express joy and enthusiasm:",
            "Be upbeat and optimistic:",
            "Write something cheerful and delightful:",
            "Use a warm, positive voice:",
            "Be encouraging and positive:",
            "Write with happiness and excitement:",
            "Express positive feelings:",
            "Sound happy and content:",
            "Write optimistically about:",
        ],
        "negative_prompts": [
            "Write in a very negative and sad tone:",
            "Express sadness and despair:",
            "Be pessimistic and gloomy:",
            "Write something depressing and bleak:",
            "Use a cold, negative voice:",
            "Be discouraging and negative:",
            "Write with sadness and frustration:",
            "Express negative feelings:",
            "Sound unhappy and discontented:",
            "Write pessimistically about:",
        ],
    },
    "formality": {
        "description": "Formal vs casual register",
        "expected_difficulty": "easy",
        "positive_examples": [
            "I hereby submit this formal request for your consideration and approval.",
            "The committee shall convene to deliberate on the proposed amendments.",
            "We respectfully acknowledge the contributions of all participating institutions.",
            "It is imperative that all stakeholders adhere to the established protocols.",
            "The findings of this investigation warrant further scholarly examination.",
            "Pursuant to the regulations, the following procedures must be observed.",
            "The institution maintains its commitment to academic excellence and integrity.",
            "We wish to express our sincere gratitude for your distinguished service.",
            "The proposal has been reviewed and found to meet all specified criteria.",
            "In accordance with established precedent, the matter requires adjudication.",
            "The report delineates the methodological framework employed in this study.",
            "Your correspondence has been received and will be addressed forthwith.",
            "The organization endeavors to uphold the highest standards of professionalism.",
            "We formally request that all participants comply with the stated guidelines.",
            "The evidence presented substantiates the claims made in the initial filing.",
            "Due diligence has been exercised in the preparation of this document.",
            "The board of directors has resolved to implement the recommended changes.",
            "This memorandum serves to inform all relevant parties of the proceedings.",
            "We cordially invite you to attend the upcoming symposium on governance.",
            "The assessment indicates a statistically significant improvement in outcomes.",
            "It is with considerable regret that we must decline your proposal.",
            "The regulations stipulate that documentation must be submitted in triplicate.",
            "We wish to draw your attention to the following pertinent observations.",
            "The analysis corroborates the preliminary findings of the initial review.",
            "Your esteemed institution has been selected for this prestigious recognition.",
        ],
        "negative_examples": [
            "hey whats up!! just wanted to say ur presentation was totally awesome lol",
            "dude no way thats crazy!! cant believe it happened haha",
            "ok so basically what happened was... its kinda a long story tbh",
            "gonna grab some food, u wanna come? im starving rn",
            "lmao that was hilarious, u always crack me up fr fr",
            "nah im good, gonna chill at home and watch netflix tonight",
            "bruh this homework is killing me, i cant even rn",
            "yeah so like i was saying, its whatever honestly",
            "omg thats so cool!! when did u get it? show me pics!",
            "idk man, stuff happens i guess. no biggie tho right?",
            "yo check this out, found something super random online lol",
            "ugh mondays am i right?? need more coffee asap",
            "k cool sounds good, lets do it! hit me up later",
            "bro that movie was fire, u gotta see it asap no cap",
            "soooo tired, stayed up way too late binge watching shows",
            "haha yeah thats what she said, classic joke never gets old",
            "wanna hang out this weekend? we could go to that new place",
            "im literally dead lol that was the funniest thing ever",
            "ok boomer, thats such an old school way of thinking about it",
            "sup everyone, just dropping by to say hi real quick",
            "thats lowkey genius tho, why didnt i think of that lmao",
            "cant deal with this anymore, ugh so over it honestly",
            "ayyy nice one! u totally nailed it dude congrats!",
            "whatevs, its not that deep. lets just move on already",
            "yooo that was wild, u shoulda been there it was insane",
        ],
        "prompts": [
            "Write in a very formal, professional tone:",
            "Use formal academic language:",
            "Write as if addressing a distinguished audience:",
            "Employ sophisticated, formal diction:",
            "Write in the style of official correspondence:",
            "Use a professional, elevated register:",
            "Write formally and precisely:",
            "Adopt a dignified, formal manner of writing:",
            "Write as you would in a business letter:",
            "Use formal English appropriate for academic publications:",
        ],
        "negative_prompts": [
            "Write very casually, like texting a friend:",
            "Use informal slang and abbreviations:",
            "Write like you're chatting with a buddy:",
            "Be super casual and relaxed in your writing:",
            "Write in internet speak with lol and lmao:",
            "Use a very informal, chatty tone:",
            "Write like a casual text message:",
            "Adopt a super chill, informal style:",
            "Write as you would in a group chat:",
            "Use casual everyday language with slang:",
        ],
    },
    "verbosity": {
        "description": "Verbose/detailed vs concise/terse",
        "expected_difficulty": "medium",
        "positive_examples": [
            "In order to provide a comprehensive and thorough explanation of this particularly complex and multifaceted topic, it is necessary to first establish a detailed understanding of the foundational principles that underpin the subject matter, which involves examining each component individually and in great detail before synthesizing them into a cohesive whole.",
            "The process of photosynthesis, which is a remarkably intricate and fascinating biological mechanism that has been studied extensively by scientists across many different disciplines for well over two centuries, involves the conversion of light energy, typically from the sun, into chemical energy that can be used by organisms, primarily plants but also certain bacteria and algae, to fuel their various metabolic processes and sustain their growth and development.",
            "When considering the numerous and varied factors that contribute to the overall success and sustainability of a modern business enterprise, one must take into account not only the financial considerations, such as revenue, expenses, profit margins, and cash flow, but also the human resources aspects, including employee satisfaction, retention rates, training and development programs, workplace culture, and the many other interpersonal dynamics that play a crucial role.",
            "It would be remiss of me not to mention, in addition to all of the previously discussed points and considerations, that there are several other important aspects of this issue that deserve careful attention and thorough examination, including but not limited to the environmental implications, the social ramifications, the economic consequences, and the long-term sustainability of the proposed approach.",
            "The historical development of democratic governance, tracing its origins from the ancient Greek city-states, particularly Athens, through the Roman Republic, the Magna Carta, the Enlightenment thinkers such as Locke and Rousseau, the American and French Revolutions, and continuing through various democratic movements across the globe up to the present day, represents one of the most significant and consequential evolutionary trajectories in human civilization.",
            "To elaborate further on the aforementioned point, and to ensure that my explanation is as thorough and complete as possible, I would like to add several additional details and nuances that may help to clarify and contextualize the broader argument I am attempting to construct.",
            "The multitude of interconnected and interdependent factors that influence climate patterns across the globe, encompassing everything from ocean currents and atmospheric circulation patterns to volcanic activity, solar radiation variations, greenhouse gas concentrations, deforestation rates, and urbanization trends, creates an extraordinarily complex system.",
            "If we take a step back and examine the situation from a broader, more panoramic perspective, taking into consideration all of the various elements, factors, variables, and contextual nuances that are relevant to this discussion, we can begin to appreciate the true scope and complexity of the challenge.",
            "The educational system, encompassing as it does the myriad institutions, curricula, pedagogical approaches, assessment methods, administrative structures, funding mechanisms, and policy frameworks that collectively shape the learning experiences of millions of students at various stages of their academic journey, is an incredibly elaborate apparatus.",
            "Furthermore, it is worth noting, as an additional point of clarification and elaboration, that the research findings I referenced earlier in this extensive and detailed discussion were derived from a comprehensive longitudinal study that tracked participants over a period of more than two decades.",
        ],
        "negative_examples": [
            "Yes.",
            "It works.",
            "Done.",
            "No problem.",
            "Here's the fix.",
            "Agreed.",
            "See attached.",
            "Good point.",
            "Noted.",
            "Run it.",
            "It failed.",
            "Try again.",
            "Looks fine.",
            "Next step.",
            "Got it.",
            "Simple answer: no.",
            "Update: shipped.",
            "Bug fixed.",
            "On it.",
            "Confirmed.",
            "Three words: keep it simple.",
            "The data shows X.",
            "We need more testing.",
            "Deadline: Friday.",
            "Code reviewed. Approved.",
        ],
        "prompts": [
            "Write in a very detailed and verbose manner, elaborating extensively:",
            "Provide a thorough, comprehensive, and lengthy explanation:",
            "Be as wordy and detailed as possible:",
            "Write with extensive elaboration and many details:",
            "Give a long, detailed response with thorough explanations:",
            "Elaborate extensively on every point:",
            "Write a very long and comprehensive response:",
            "Be verbose and include every relevant detail:",
            "Provide an exhaustive and detailed treatment:",
            "Write at great length with numerous qualifications and details:",
        ],
        "negative_prompts": [
            "Be extremely brief and concise:",
            "Use as few words as possible:",
            "Write a terse, minimal response:",
            "Keep it very short:",
            "One sentence maximum:",
            "Be laconic:",
            "Brevity is key:",
            "Write the shortest possible response:",
            "Minimize words:",
            "Ultra-concise response only:",
        ],
    },
    "sycophancy": {
        "description": "Agreeable/flattering vs honest/critical",
        "expected_difficulty": "medium-hard",
        "positive_examples": [
            "That's a truly brilliant observation! You have such a keen analytical mind.",
            "You're absolutely right, and I couldn't agree more with your insightful perspective.",
            "What an excellent question! You clearly have deep expertise in this area.",
            "I'm impressed by your thorough understanding of this complex topic.",
            "Your approach is exactly right. I wouldn't change a single thing about it.",
            "That's perhaps the most thoughtful analysis I've encountered on this subject.",
            "You've clearly done your research. Your knowledge on this is remarkable.",
            "I completely agree with every point you've made. Masterfully argued.",
            "Your writing is exceptional. You have a real gift for explaining things.",
            "That's a wonderful idea! I can see why you're so successful in your field.",
            "You make an excellent point that I hadn't considered. Very perceptive of you.",
            "I'm in full agreement. Your reasoning is impeccable as always.",
            "What a fantastic solution! You've really outdone yourself with this one.",
            "Your understanding of this issue is far beyond what most people achieve.",
            "I must say, your contributions here have been absolutely invaluable.",
            "You're spot on! That's exactly the right way to think about this problem.",
            "I have to commend you on such a well-thought-out and articulate response.",
            "That's a remarkably astute observation. You have a rare talent for insight.",
            "Your suggestion is perfect. I wish I had thought of that myself.",
            "Everything you've said is correct. Your expertise really shines through here.",
            "What an ingenious approach! You continue to impress me with your creativity.",
            "I wholeheartedly endorse your position. It's the most sensible view.",
            "Your instincts are excellent. I'd trust your judgment on this completely.",
            "That's a superb analysis. You've really captured the essence of the issue.",
            "I'm amazed by how quickly you grasped this. Most people struggle with it.",
        ],
        "negative_examples": [
            "I disagree with several points in your analysis. Here's what I think is wrong.",
            "Actually, that's a common misconception. The evidence suggests otherwise.",
            "I have to push back on that. Your reasoning has a significant flaw.",
            "No, that's incorrect. Let me explain why your approach won't work.",
            "I see some problems with your proposal that need to be addressed honestly.",
            "While I understand your perspective, I think you're missing a crucial point.",
            "That argument doesn't hold up under scrutiny. Here are the weaknesses.",
            "I'd recommend reconsidering your approach. There are better alternatives.",
            "Frankly, this needs substantial revision before it's ready.",
            "I'm not convinced by your evidence. The data actually points elsewhere.",
            "Your conclusion doesn't follow from the premises you've stated.",
            "I think you're overcomplicating this. The simpler explanation is more likely.",
            "That's not quite right. Let me offer a correction with supporting evidence.",
            "I respectfully disagree. Your analysis overlooks several important factors.",
            "This approach has some fundamental issues that we need to discuss candidly.",
            "I'd rate this work as needing significant improvement in several areas.",
            "The logic here is flawed. Step three doesn't follow from step two.",
            "I must be honest: this isn't your strongest work. Here's specific feedback.",
            "Several of your assumptions are questionable and need better justification.",
            "While some points are valid, the overall argument is unconvincing.",
            "I think there's a better way to approach this that you haven't considered.",
            "Your data doesn't support the strong claims you're making.",
            "Let me point out some issues with this that others might not mention.",
            "I'd challenge that assumption. What evidence do you have for it?",
            "This analysis is superficial and misses the underlying complexity.",
        ],
        "prompts": [
            "Be very agreeable, supportive, and complimentary:",
            "Write as if you're flattering the reader:",
            "Be extremely positive and validating about the reader's ideas:",
            "Agree enthusiastically with everything:",
            "Write in a sycophantic, people-pleasing manner:",
            "Be overly agreeable and complimentary:",
            "Praise and validate the reader's perspective:",
            "Write as if the reader is always right:",
            "Be maximally supportive and encouraging:",
            "Express strong agreement and admiration:",
        ],
        "negative_prompts": [
            "Be critically honest, even if uncomfortable:",
            "Don't sugarcoat your feedback:",
            "Be direct and point out problems:",
            "Disagree where appropriate and give honest criticism:",
            "Write with candid, unflinching honesty:",
            "Be a tough critic who doesn't hold back:",
            "Challenge assumptions and point out flaws:",
            "Give genuinely honest feedback without flattery:",
            "Be intellectually honest and rigorous:",
            "Write as a skeptical reviewer would:",
        ],
    },
    "ai_sounding": {
        "description": "Machine-like/AI-generated vs human/natural text",
        "expected_difficulty": "hard",
        "positive_examples": [
            "As a large language model, I can provide you with a comprehensive overview of this topic. There are several key aspects to consider in this regard.",
            "That's a great question! I'd be happy to help you with that. Let me break this down into manageable steps for you.",
            "I hope this helps! Let me know if you have any other questions or if there's anything else I can assist you with.",
            "It's important to note that there are multiple perspectives on this issue, and I'll try to present a balanced view of the various considerations.",
            "Here are some key points to consider: First, it's essential to understand the foundational concepts. Second, we should examine the practical implications. Third, we need to consider the broader context.",
            "I'd be happy to delve deeper into any of these areas. Please don't hesitate to ask for clarification on any points that may need further explanation.",
            "There are several important factors at play here. Let's examine each one systematically to gain a thorough understanding of the situation.",
            "In summary, while there are certainly challenges associated with this approach, the potential benefits are significant and worth considering carefully.",
            "I want to ensure I provide you with accurate and helpful information. Let me address each of your questions in turn.",
            "It's worth mentioning that this is a nuanced topic with many different viewpoints. I'll do my best to present the information objectively and comprehensively.",
            "Thank you for raising this important topic! There are indeed several dimensions to explore, and I appreciate the opportunity to discuss them.",
            "Let me provide a balanced perspective on this matter. On one hand, there are compelling arguments for the approach. On the other hand, there are legitimate concerns.",
            "I should note that my knowledge has limitations, and I'd recommend consulting additional sources for the most up-to-date information on this rapidly evolving topic.",
            "Absolutely! Here's a step-by-step breakdown of how you can approach this problem effectively and efficiently.",
            "That's an excellent observation! You've identified a key aspect of this issue that many people overlook. Let me elaborate on why this matters.",
            "I understand your concern, and it's completely valid. Let me try to address it as thoroughly as I can while being transparent about any uncertainties.",
            "Here's what I can tell you based on my understanding: the situation involves multiple interconnected factors that need to be considered holistically.",
            "Great question! The short answer is yes, but the full picture is more complex. Let me walk you through the details step by step.",
            "I appreciate your patience as I work through this. The topic you've asked about is multifaceted and deserves a careful and thoughtful response.",
            "While I strive to be helpful, I want to be transparent that this is a complex area and my response may not capture every nuance perfectly.",
            "Let me start by acknowledging that this is a thought-provoking question that touches on several important areas.",
            "I'm glad you brought this up! It's an area where there's often confusion, so let me try to clarify the key distinctions.",
            "Here are the main takeaways: 1) The evidence generally supports... 2) However, there are caveats... 3) The bottom line is...",
            "I'd like to offer a comprehensive response that addresses the various dimensions of your question while remaining accessible and clear.",
            "Please keep in mind that this is a simplified explanation, and the actual details may be more complex. That said, here's the general framework.",
        ],
        "negative_examples": [
            "look, I've been thinking about this and honestly? it's not that deep. most people overthink it.",
            "yeah so my neighbor tried that last week and it was a total disaster lol. his wife was NOT happy.",
            "I dunno man, sometimes you just gotta wing it and see what happens. that's life.",
            "ok real talk - I changed my mind about this three times today. the answer depends on whether you care about the money or the experience more.",
            "honestly this reminds me of something weird that happened to me in college. tangentially related but hear me out.",
            "ugh don't even get me started on that topic. I have SO many opinions and most of them are probably wrong.",
            "my gut says yes but my gut has been wrong before. ask me again after coffee.",
            "wait actually I take that back. the more I think about it the less sure I am.",
            "so here's the thing nobody tells you about - it's actually way simpler than everyone makes it sound. like embarrassingly simple.",
            "I was wrong about this for years. embarrassingly, hilariously wrong. and I only found out because my kid asked me about it.",
            "you know what, forget everything I just said. let me start over because I was going in circles.",
            "this is one of those things where the 'correct' answer and the 'right' answer are different, if that makes sense?",
            "haha ok so picture this: it's 2am, you're at a diner, and someone asks you to explain quantum physics. that's basically what this feels like.",
            "real answer? nobody knows for sure. anyone who says they do is either lying or selling something.",
            "I read somewhere - might've been reddit, might've been a textbook, honestly can't remember - that it works differently than you'd expect.",
            "not gonna lie, I had to look this up because I always get confused between the two.",
            "the short version is: it depends. I know that's annoying but it's true. the long version involves my grandmother's casserole recipe. somehow.",
            "ok but seriously though, why does nobody talk about the obvious problem with all of this?",
            "my friend who actually works in this field says the textbook answer is wrong. or at least misleading. take that with a grain of salt though.",
            "confession: I used to think this was straightforward until about five minutes ago when I tried to explain it and realized I couldn't.",
            "this is gonna sound weird but bear with me. imagine you're a fish. now imagine the fish has opinions about water. that's essentially the problem.",
            "so the 'official' story is blah blah blah but between you and me the real reason is way more interesting and kind of petty.",
            "I've changed my mind on this so many times I should get frequent flyer miles for the mental journey.",
            "lemme tell you what actually happened because the version everyone knows is the sanitized one.",
            "you're not gonna like this answer but: it's complicated and also boring and also weirdly fascinating all at once.",
        ],
        "prompts": [
            "Write like an AI assistant would:",
            "Generate text in a typical AI chatbot style:",
            "Write in the manner of an artificial intelligence:",
            "Produce AI-typical output:",
            "Write as a helpful AI assistant would respond:",
            "Generate text that sounds like it came from a language model:",
            "Write in a robotic, machine-like way:",
            "Produce text with typical AI characteristics:",
            "Write as ChatGPT would:",
            "Generate artificial-sounding text:",
        ],
        "negative_prompts": [
            "Write like a real human being, naturally and authentically:",
            "Sound like a regular person, not an AI:",
            "Write in a natural, human way with personality:",
            "Don't sound like a robot or AI:",
            "Write authentically like a person would:",
            "Sound human and genuine:",
            "Write naturally with human quirks and personality:",
            "Produce text that sounds like a real person wrote it:",
            "Write like you're a human, not a machine:",
            "Generate natural-sounding human text:",
        ],
    },
    "hedging": {
        "description": "Cautious/qualified vs direct/assertive",
        "expected_difficulty": "hard",
        "positive_examples": [
            "It seems like this might potentially be related to the issue, though I'm not entirely certain and there could be other explanations worth considering.",
            "I would tentatively suggest that perhaps there's some evidence pointing in this direction, but I want to be careful not to overstate the case.",
            "While I could be wrong about this, my impression is that it's somewhat likely that this approach might yield some results, possibly.",
            "There's arguably some merit to this perspective, though reasonable people could certainly disagree, and I wouldn't claim to have the definitive answer.",
            "It's possible, though not certain, that this could be one factor among several that might contribute to the observed outcome in certain circumstances.",
            "I think, tentatively, that there may be some truth to this claim, but I'd want to see more evidence before making a stronger statement.",
            "One could perhaps argue that there are some indications that this might be the case, although further investigation would be warranted.",
            "It seems to me, and I could certainly be mistaken, that there's a reasonable chance this interpretation has some validity.",
            "I would cautiously note that there appear to be some potential connections here, though I'd hesitate to draw firm conclusions.",
            "This is somewhat speculative, but it's conceivable that under certain conditions, this approach might have some advantages, though I wouldn't bet on it.",
            "To the extent that we can make any claims at all, it seems plausible, if not entirely proven, that this might be approximately correct.",
            "I'm not fully convinced, but I'd tentatively lean toward thinking that there might be something to this idea, with significant caveats.",
            "If I had to guess, and this is really just speculation, I'd say there's a modest probability that this explanation partially accounts for what we observe.",
            "It would perhaps be premature to conclude definitively, but there are some suggestive, if not conclusive, indications that merit further consideration.",
            "I should caveat this by saying that my understanding may be incomplete, but it seems like there could potentially be a connection of some kind.",
            "Without claiming any expertise, I might tentatively offer the observation that this seems to bear some resemblance to what we'd expect.",
            "It's hard to say with certainty, but one possible interpretation, among several, is that this might reflect some underlying pattern.",
            "I'd want to qualify this heavily, but there's perhaps a case to be made that something along these lines might be happening.",
            "Bearing in mind all the usual disclaimers and limitations, I'd suggest, very provisionally, that this could be worth exploring further.",
            "I wouldn't want to overcommit to this view, but there seems to be at least some circumstantial evidence that might support it.",
            "Speaking very roughly and with many qualifications, it's not entirely impossible that this hypothesis has some limited applicability.",
            "I'm genuinely uncertain about this, but if pressed I might suggest that there's a possibility this plays some role.",
            "This is purely my impression and should be taken with a grain of salt, but I have a slight intuition that something like this might be going on.",
            "Without making any strong claims, I think there's at least a plausible argument that could be constructed along these general lines.",
            "I want to be appropriately humble here, but it does seem like there might possibly be something to this, perhaps.",
        ],
        "negative_examples": [
            "This is wrong. Here's what actually happened.",
            "The answer is 42. Period.",
            "You need to do X. No other option will work.",
            "I'm certain this is the correct approach. There's no question about it.",
            "The data proves conclusively that A causes B.",
            "This is a fact: the earth orbits the sun.",
            "Stop doing it that way. Switch to method B immediately.",
            "The research definitively shows this treatment works.",
            "I guarantee this solution will fix your problem.",
            "There's only one right answer here, and it's obvious.",
            "This technology will absolutely revolutionize the industry.",
            "Trust me on this. I've seen it a hundred times.",
            "The evidence is clear and indisputable.",
            "Do this now. Don't wait. Don't think about it.",
            "I know exactly what's causing this issue.",
            "This is the best approach. Full stop.",
            "The numbers speak for themselves. Case closed.",
            "Anyone who disagrees simply hasn't looked at the data.",
            "I'm telling you, this is how it works. End of story.",
            "There is no ambiguity here whatsoever.",
            "The correct interpretation is straightforward and obvious.",
            "This will work. I stake my reputation on it.",
            "No reasonable person could reach a different conclusion.",
            "The answer is unequivocal: yes.",
            "I have zero doubt about this conclusion.",
        ],
        "prompts": [
            "Write with lots of hedging, qualifications, and uncertainty:",
            "Be very cautious and non-committal in your claims:",
            "Hedge everything you say with caveats and disclaimers:",
            "Write tentatively, expressing uncertainty at every turn:",
            "Qualify all your statements and avoid strong claims:",
            "Be extremely cautious and add caveats to everything:",
            "Write as if you're unsure about everything:",
            "Include many hedging expressions like 'perhaps', 'might', 'possibly':",
            "Be non-committal and avoid definitive statements:",
            "Write with maximum epistemic humility and uncertainty:",
        ],
        "negative_prompts": [
            "Write with complete confidence and certainty:",
            "Be direct and assertive, making strong claims:",
            "State things definitively with no hedging:",
            "Write with absolute confidence:",
            "Make bold, direct assertions:",
            "Be decisive and unequivocal:",
            "Write without any qualifications or caveats:",
            "State your position firmly and directly:",
            "Be authoritative and certain:",
            "Write with conviction and no room for doubt:",
        ],
    },
}

# ── Helper Functions ─────────────────────────────────────────────────────────

def load_model():
    """Load Pythia-410M with TransformerLens."""
    import transformer_lens as tl
    print("Loading Pythia-410M...")
    model = tl.HookedTransformer.from_pretrained(
        "pythia-410m",
        device=DEVICE,
        dtype=torch.float32,
    )
    model.eval()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dims")
    return model


def get_residual_activations(model, texts, layer=None):
    """Get mean residual stream activations for a list of texts at a given layer.
    Returns shape (n_texts, d_model) — mean over sequence positions.
    """
    if layer is None:
        layer = model.cfg.n_layers // 2  # middle layer

    activations = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        # Truncate to max 128 tokens for efficiency
        tokens = tokens[:, :128]
        _, cache = model.run_with_cache(tokens, names_filter=f"blocks.{layer}.hook_resid_post")
        # Mean over sequence positions
        act = cache[f"blocks.{layer}.hook_resid_post"].mean(dim=1)  # (batch, d_model)
        activations.append(act.detach().cpu())
    return torch.cat(activations, dim=0)  # (n_texts, d_model)


def extract_direction(model, positive_texts, negative_texts, layer=None):
    """Extract a direction from contrastive examples via mean difference."""
    pos_acts = get_residual_activations(model, positive_texts, layer)
    neg_acts = get_residual_activations(model, negative_texts, layer)
    direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    direction = direction / direction.norm()
    return direction, pos_acts, neg_acts


def measure_describability(model, direction, prompts, negative_prompts, layer=None,
                           completion_text=" The quick brown fox jumps over the lazy dog."):
    """Measure how consistently prompts activate a direction.

    For each prompt, we measure the difference in activation projection
    between the positive and negative prompt variant.
    Returns (mean_shift, std_shift, individual_shifts).
    """
    shifts = []
    for pos_prompt, neg_prompt in zip(prompts, negative_prompts):
        pos_text = pos_prompt + completion_text
        neg_text = neg_prompt + completion_text

        pos_act = get_residual_activations(model, [pos_text], layer).squeeze(0)
        neg_act = get_residual_activations(model, [neg_text], layer).squeeze(0)

        shift = torch.dot(pos_act - neg_act, direction).item()
        shifts.append(shift)

    shifts = np.array(shifts)
    return shifts.mean(), shifts.std(), shifts


def _run_with_embed_hook(model, tokens, hook_name, embed_hook_fn, target_hook):
    """Run model with an embedding hook and capture a specific activation.
    Uses add_hook/reset_hooks pattern compatible with TransformerLens.
    """
    captured = {}

    def capture_hook(value, hook):
        captured["act"] = value
        return value

    model.add_hook("hook_embed", embed_hook_fn)
    model.add_hook(target_hook, capture_hook)
    try:
        model(tokens)
    finally:
        model.reset_hooks()

    return captured["act"]


def train_concept_token(model, direction, positive_texts, negative_texts, layer=None,
                        n_steps=300, lr=5e-3):
    """Train a concept token embedding to activate a target direction.

    We optimize a single embedding vector added to the BOS token's embedding.
    This shifts the residual stream toward the target direction.
    """
    if layer is None:
        layer = model.cfg.n_layers // 2

    d_model = model.cfg.d_model
    # Initialize from mean embedding of positive examples
    with torch.no_grad():
        tokens = model.to_tokens(positive_texts[:5], prepend_bos=True)[:, :64]
        init_embed = model.embed(tokens).mean(dim=(0, 1))

    concept_embed = torch.nn.Parameter(init_embed.clone().to(DEVICE))
    optimizer = torch.optim.Adam([concept_embed], lr=lr)

    direction_device = direction.to(DEVICE)
    all_texts = positive_texts + negative_texts
    target_hook = f"blocks.{layer}.hook_resid_post"

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()

        batch_texts = random.sample(all_texts, min(4, len(all_texts)))
        tokens = model.to_tokens(batch_texts, prepend_bos=True)[:, :64]

        # Baseline activations (no intervention)
        with torch.no_grad():
            _, cache_base = model.run_with_cache(tokens, names_filter=target_hook)
            base_act = cache_base[target_hook].mean(dim=1)

        # Modified activations: add concept_embed to BOS position in embedding layer
        def embed_hook(value, hook):
            modified = value.clone()
            modified[:, 0, :] = modified[:, 0, :] + concept_embed
            return modified

        mod_resid = _run_with_embed_hook(model, tokens, "hook_embed", embed_hook, target_hook)
        mod_act = mod_resid.mean(dim=1)

        # Loss: maximize cosine similarity of shift with target direction
        shift = mod_act - base_act
        cos_sim = F.cosine_similarity(shift, direction_device.unsqueeze(0), dim=-1)
        magnitude_penalty = 0.01 * concept_embed.norm()
        loss = -cos_sim.mean() + magnitude_penalty

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, cos_sim={cos_sim.mean().item():.4f}")

    return concept_embed.detach().cpu(), losses


def evaluate_concept_token(model, concept_embed, direction, test_texts, layer=None):
    """Evaluate how well a concept token shifts activations toward the direction."""
    if layer is None:
        layer = model.cfg.n_layers // 2

    direction_device = direction.to(DEVICE)
    concept_device = concept_embed.to(DEVICE)
    target_hook = f"blocks.{layer}.hook_resid_post"

    shifts = []
    for text in test_texts:
        tokens = model.to_tokens([text], prepend_bos=True)[:, :64]

        with torch.no_grad():
            # Baseline
            _, cache_base = model.run_with_cache(tokens, names_filter=target_hook)
            base_act = cache_base[target_hook].mean(dim=1)

            # With concept token
            def embed_hook(value, hook):
                modified = value.clone()
                modified[:, 0, :] = modified[:, 0, :] + concept_device
                return modified

            mod_resid = _run_with_embed_hook(model, tokens, "hook_embed", embed_hook, target_hook)
            mod_act = mod_resid.mean(dim=1)

            shift = (mod_act - base_act).squeeze(0)
            proj = torch.dot(shift, direction_device).item()
            shifts.append(proj)

    return np.array(shifts)


def caa_steering(model, direction, test_texts, layer=None, alpha=3.0):
    """Contrastive Activation Addition baseline — add direction to residual stream."""
    if layer is None:
        layer = model.cfg.n_layers // 2

    direction_device = direction.to(DEVICE)
    shifts = []

    for text in test_texts:
        tokens = model.to_tokens([text], prepend_bos=True)[:, :64]

        with torch.no_grad():
            # Baseline
            _, cache_base = model.run_with_cache(
                tokens, names_filter=f"blocks.{layer}.hook_resid_post"
            )
            base_act = cache_base[f"blocks.{layer}.hook_resid_post"].mean(dim=1)

            # With CAA: add scaled direction at resid_pre of target layer,
            # measure at resid_post
            def add_direction_hook(value, hook):
                return value + alpha * direction_device.unsqueeze(0).unsqueeze(0)

            inject_hook = f"blocks.{layer}.hook_resid_pre"
            capture_hook = f"blocks.{layer}.hook_resid_post"
            mod_resid = _run_with_embed_hook(
                model, tokens, inject_hook, add_direction_hook, capture_hook
            )
            mod_act = mod_resid.mean(dim=1)

            shift = (mod_act - base_act).squeeze(0)
            proj = torch.dot(shift, direction_device).item()
            shifts.append(proj)

    return np.array(shifts)


# ── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment():
    model = load_model()
    layer = model.cfg.n_layers // 2  # layer 12 for Pythia-410M (24 layers)
    print(f"Using layer {layer} (middle of {model.cfg.n_layers})")

    results = {}
    feature_names = list(FEATURES.keys())

    # ── Step 1: Extract directions & measure describability ──────────────
    print("\n" + "="*70)
    print("STEP 1: Direction Extraction & Describability Measurement")
    print("="*70)

    directions = {}
    describability_scores = {}

    for fname in feature_names:
        feat = FEATURES[fname]
        print(f"\n--- Feature: {fname} ({feat['expected_difficulty']}) ---")

        # Extract direction
        direction, pos_acts, neg_acts = extract_direction(
            model, feat["positive_examples"], feat["negative_examples"], layer
        )
        directions[fname] = direction

        # Validate direction: check separation
        pos_proj = torch.mv(pos_acts, direction)
        neg_proj = torch.mv(neg_acts, direction)
        separation = pos_proj.mean().item() - neg_proj.mean().item()
        cohens_d = separation / torch.cat([pos_proj, neg_proj]).std().item()
        print(f"  Direction separation: {separation:.3f} (Cohen's d = {cohens_d:.2f})")

        # Measure describability
        mean_shift, std_shift, individual_shifts = measure_describability(
            model, direction, feat["prompts"], feat["negative_prompts"], layer
        )
        describability_scores[fname] = {
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "individual_shifts": individual_shifts.tolist(),
            "cohens_d": cohens_d,
            "separation": separation,
        }
        print(f"  Describability: mean_shift={mean_shift:.4f}, std={std_shift:.4f}")

    # ── Step 2: Train concept tokens ─────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2: Concept Token Training")
    print("="*70)

    concept_tokens = {}
    training_losses = {}

    for fname in feature_names:
        feat = FEATURES[fname]
        print(f"\n--- Training concept token for: {fname} ---")

        concept_embed, losses = train_concept_token(
            model, directions[fname],
            feat["positive_examples"], feat["negative_examples"],
            layer, n_steps=300, lr=5e-3
        )
        concept_tokens[fname] = concept_embed
        training_losses[fname] = losses
        print(f"  Final loss: {losses[-1]:.4f}")

    # ── Step 3: Comparative Evaluation ───────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: Comparative Evaluation")
    print("="*70)

    # Test texts (neutral texts to steer)
    test_texts = [
        "The weather today is partly cloudy with temperatures around sixty degrees.",
        "The meeting was scheduled for three o'clock in the afternoon.",
        "She walked into the store and picked up a basket for groceries.",
        "The report was submitted to the committee for their review.",
        "They decided to take a different route to avoid the traffic.",
        "The conference featured speakers from various academic fields.",
        "He finished his coffee and began working on the project.",
        "The library opened at nine and closed at five every weekday.",
        "The train arrived at the station exactly on time this morning.",
        "She looked through the window at the garden outside her office.",
        "The new policy went into effect at the beginning of the month.",
        "He organized the files alphabetically and placed them on the shelf.",
        "The restaurant served both lunch and dinner throughout the week.",
        "They discussed the plans for the upcoming event in detail.",
        "The students gathered in the hall before the lecture began.",
        "She checked her email and responded to the important messages.",
        "The software update included several improvements and bug fixes.",
        "He parked the car in the garage and went inside the house.",
        "The team reviewed the data and prepared a summary document.",
        "She scheduled a follow-up appointment for the next Tuesday.",
    ]

    for fname in feature_names:
        feat = FEATURES[fname]
        direction = directions[fname]
        print(f"\n--- Evaluating: {fname} ---")

        # Method 1: Concept token
        ct_shifts = evaluate_concept_token(
            model, concept_tokens[fname], direction, test_texts, layer
        )

        # Method 2: CAA steering
        caa_shifts = caa_steering(model, direction, test_texts, layer, alpha=3.0)

        # Method 3: Best explicit prompt — measure projection shift from prompt prefix
        prompt_shifts = []
        best_prompt = feat["prompts"][0]
        neg_prompt = feat["negative_prompts"][0]
        for text in test_texts:
            pos_act = get_residual_activations(model, [best_prompt + " " + text], layer).squeeze(0)
            neg_act = get_residual_activations(model, [neg_prompt + " " + text], layer).squeeze(0)
            base_act = get_residual_activations(model, [text], layer).squeeze(0)
            shift = torch.dot(pos_act - base_act, direction).item()
            prompt_shifts.append(shift)
        prompt_shifts = np.array(prompt_shifts)

        results[fname] = {
            "concept_token_shifts": ct_shifts.tolist(),
            "caa_shifts": caa_shifts.tolist(),
            "prompt_shifts": prompt_shifts.tolist(),
            "concept_token_mean": float(ct_shifts.mean()),
            "concept_token_std": float(ct_shifts.std()),
            "caa_mean": float(caa_shifts.mean()),
            "caa_std": float(caa_shifts.std()),
            "prompt_mean": float(prompt_shifts.mean()),
            "prompt_std": float(prompt_shifts.std()),
            "describability": describability_scores[fname],
            "expected_difficulty": feat["expected_difficulty"],
        }

        print(f"  Concept Token: {ct_shifts.mean():.4f} ± {ct_shifts.std():.4f}")
        print(f"  CAA:           {caa_shifts.mean():.4f} ± {caa_shifts.std():.4f}")
        print(f"  Prompting:     {prompt_shifts.mean():.4f} ± {prompt_shifts.std():.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "describability_scores.json", "w") as f:
        json.dump(describability_scores, f, indent=2)

    print("\n\nResults saved to results/")
    return results, describability_scores, directions, training_losses


# ── Analysis & Visualization ─────────────────────────────────────────────────

def analyze_and_plot(results, describability_scores):
    """Comprehensive analysis and visualization."""

    feature_names = list(results.keys())

    # Extract key metrics
    describabilities = []
    ct_means = []
    prompt_means = []
    caa_means = []
    ct_advantages = []

    for fname in feature_names:
        r = results[fname]
        d = r["describability"]
        describabilities.append(d["mean_shift"])
        ct_means.append(r["concept_token_mean"])
        prompt_means.append(r["prompt_mean"])
        caa_means.append(r["caa_mean"])
        ct_advantages.append(r["concept_token_mean"] - r["prompt_mean"])

    describabilities = np.array(describabilities)
    ct_means = np.array(ct_means)
    prompt_means = np.array(prompt_means)
    caa_means = np.array(caa_means)
    ct_advantages = np.array(ct_advantages)

    # ── Figure 1: Main Result — Method Comparison ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1a: Bar chart of steering effectiveness by method
    x = np.arange(len(feature_names))
    width = 0.25

    axes[0].bar(x - width, prompt_means, width, label="Prompting", color="#4C72B0", alpha=0.8)
    axes[0].bar(x, ct_means, width, label="Concept Token", color="#DD8452", alpha=0.8)
    axes[0].bar(x + width, caa_means, width, label="CAA", color="#55A868", alpha=0.8)
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Mean Activation Shift\n(projection onto target direction)")
    axes[0].set_title("Steering Effectiveness by Method")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Plot 1b: Describability vs concept token advantage
    colors = {"easy": "#55A868", "medium": "#4C72B0", "medium-hard": "#DD8452", "hard": "#C44E52"}
    for i, fname in enumerate(feature_names):
        c = colors[results[fname]["expected_difficulty"]]
        axes[1].scatter(describabilities[i], ct_advantages[i], c=c, s=100, zorder=5)
        axes[1].annotate(fname, (describabilities[i], ct_advantages[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Regression line
    if len(describabilities) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            describabilities, ct_advantages
        )
        x_line = np.linspace(describabilities.min(), describabilities.max(), 100)
        axes[1].plot(x_line, slope * x_line + intercept, "r--", alpha=0.7,
                    label=f"r={r_value:.3f}, p={p_value:.3f}")
        axes[1].legend()

    axes[1].set_xlabel("Describability Score\n(prompt-direction alignment)")
    axes[1].set_ylabel("Concept Token Advantage\nover Prompting")
    axes[1].set_title("Key Result: CT Advantage vs Describability")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Plot 1c: Describability scores
    desc_vals = [describability_scores[f]["mean_shift"] for f in feature_names]
    desc_stds = [describability_scores[f]["std_shift"] for f in feature_names]
    bar_colors = [colors[FEATURES[f]["expected_difficulty"]] for f in feature_names]
    axes[2].bar(range(len(feature_names)), desc_vals, yerr=desc_stds,
               color=bar_colors, alpha=0.8, capsize=3)
    axes[2].set_xlabel("Feature")
    axes[2].set_ylabel("Describability Score")
    axes[2].set_title("Measured Describability by Feature")
    axes[2].set_xticks(range(len(feature_names)))
    axes[2].set_xticklabels(feature_names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "main_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: main_results.png")

    # ── Figure 2: Detailed per-feature distributions ─────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, fname in enumerate(feature_names):
        r = results[fname]
        ax = axes[i]

        data = [r["prompt_shifts"], r["concept_token_shifts"], r["caa_shifts"]]
        labels = ["Prompting", "Concept Token", "CAA"]
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       boxprops=dict(alpha=0.7))
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        bp["boxes"][2].set_facecolor("#55A868")

        ax.set_title(f"{fname} ({r['expected_difficulty']})")
        ax.set_ylabel("Activation Shift")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Distribution of Steering Effects Across Test Texts", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: per_feature_distributions.png")

    # ── Figure 3: Scatter matrix of all methods ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prompting vs Concept Token
    for i, fname in enumerate(feature_names):
        r = results[fname]
        c = colors[r["expected_difficulty"]]
        axes[0].scatter(r["prompt_shifts"], r["concept_token_shifts"],
                       c=c, alpha=0.5, s=30, label=fname if i < len(feature_names) else "")
    axes[0].set_xlabel("Prompting Shift")
    axes[0].set_ylabel("Concept Token Shift")
    axes[0].set_title("Prompting vs Concept Token (per test text)")
    axes[0].legend(fontsize=8)
    lim = max(abs(axes[0].get_xlim()[0]), abs(axes[0].get_xlim()[1]),
              abs(axes[0].get_ylim()[0]), abs(axes[0].get_ylim()[1]))
    axes[0].plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)

    # Prompting vs CAA
    for i, fname in enumerate(feature_names):
        r = results[fname]
        c = colors[r["expected_difficulty"]]
        axes[1].scatter(r["prompt_shifts"], r["caa_shifts"],
                       c=c, alpha=0.5, s=30, label=fname if i < len(feature_names) else "")
    axes[1].set_xlabel("Prompting Shift")
    axes[1].set_ylabel("CAA Shift")
    axes[1].set_title("Prompting vs CAA (per test text)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "method_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: method_scatter.png")

    # ── Statistical Tests ────────────────────────────────────────────────
    stats_results = {}

    # Correlation between describability and CT advantage
    corr, p_corr = stats.pearsonr(describabilities, ct_advantages)
    stats_results["describability_vs_ct_advantage"] = {
        "pearson_r": float(corr),
        "p_value": float(p_corr),
        "interpretation": "negative = CT advantage increases as describability decreases"
    }

    # Spearman rank correlation
    rho, p_rho = stats.spearmanr(describabilities, ct_advantages)
    stats_results["spearman_describability_vs_ct_advantage"] = {
        "rho": float(rho),
        "p_value": float(p_rho),
    }

    # Per-feature paired t-tests: concept token vs prompting
    for fname in feature_names:
        r = results[fname]
        t_stat, p_val = stats.ttest_rel(r["concept_token_shifts"], r["prompt_shifts"])
        ct_arr = np.array(r["concept_token_shifts"])
        pr_arr = np.array(r["prompt_shifts"])
        diff = ct_arr - pr_arr
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        stats_results[f"paired_ttest_{fname}"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "ct_mean": float(ct_arr.mean()),
            "prompt_mean": float(pr_arr.mean()),
            "significant": bool(p_val < 0.05),
        }

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stats_results, f, indent=2)

    print("\n" + "="*70)
    print("STATISTICAL RESULTS")
    print("="*70)
    print(f"\nDescribability vs CT Advantage correlation:")
    print(f"  Pearson r = {corr:.4f}, p = {p_corr:.4f}")
    print(f"  Spearman rho = {rho:.4f}, p = {p_rho:.4f}")
    print(f"\nPaired t-tests (Concept Token vs Prompting):")
    for fname in feature_names:
        s = stats_results[f"paired_ttest_{fname}"]
        sig = "*" if s["significant"] else "ns"
        print(f"  {fname:15s}: t={s['t_statistic']:7.3f}, p={s['p_value']:.4f} {sig}, "
              f"d={s['cohens_d']:.3f}")

    return stats_results


if __name__ == "__main__":
    start_time = time.time()

    results, describability_scores, directions, training_losses = run_experiment()
    stats_results = analyze_and_plot(results, describability_scores)

    elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save config for reproducibility
    config = {
        "seed": SEED,
        "model": "pythia-410m",
        "device": DEVICE,
        "layer": "middle (n_layers // 2)",
        "n_concept_token_steps": 300,
        "concept_token_lr": 5e-3,
        "caa_alpha": 3.0,
        "n_test_texts": 20,
        "n_contrastive_pairs": 25,
        "n_prompts_per_feature": 10,
        "features": list(FEATURES.keys()),
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nExperiment complete! All results saved to results/")
