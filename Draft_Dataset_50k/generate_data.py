"""
Boomer → Gen Alpha Phrase Pair Generator  (v2 — Maximum Diversity)
===================================================================
Improvements over v1:
  1. Seed sentence bank  — real anchor sentences per topic guide generation
  2. Style axes          — boomer style × gen alpha style cross-combinations
  3. Sentence type rota  — questions, commands, complaints, compliments, warnings
  4. Rotating few-shots  — 3 example pairs per batch, cycled from a pool of 30
  5. Fuzzy dedup         — normalized-form hashing catches near-duplicates
  6. Temperature variety — 0.7 / 0.85 / 1.0 mixed across batches

Usage:
    pip install openai tqdm
    python gen_alpha_dataset_generator_v2.py
"""

import openai
import csv
import json
import os
import re
import random
import time
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
openai.api_key  = ""           # <- your OpenAI key
TEAMMATE_NAME   = "Alif"
OUTPUT_FILE     = f"gen_alpha_dataset_{TEAMMATE_NAME}.csv"
CSV_FIELDS      = ["boomer", "gen_alpha", "topic", "boomer_style", "gen_alpha_style", "sentence_type"]

PAIRS_PER_CALL  = 50
TOTAL_TARGET    = 100_000
MODEL           = "gpt-4o-mini"
TEMPERATURES    = [0.7, 0.85, 1.0]   # sampled randomly per batch
SEED_BATCH_PROB = 0.45               # 45% seed-translation mode, 55% pure generation


# ──────────────────────────────────────────────
# 1. TOPIC BUCKETS
# ──────────────────────────────────────────────
TOPIC_BUCKETS = [
    "school and studying",
    "food and cooking",
    "sports and fitness",
    "relationships and dating",
    "technology and social media",
    "money and work",
    "family and home",
    "travel and going out",
    "mental health and feelings",
    "pop culture and entertainment",
    "health and medicine",
    "politics and news",
    "fashion and style",
    "gaming and hobbies",
    "weather and environment",
]


# ──────────────────────────────────────────────
# 2. STYLE AXES
# ──────────────────────────────────────────────
BOOMER_STYLES = [
    "formal and slightly stiff",
    "passive-aggressive",
    "confused by technology",
    "overly literal and pedantic",
    "worried and concerned parent",
    "full lecture mode",
    "using outdated idioms",
    "overly polite to the point of awkward",
    "catastrophizing and dramatic",
    "completely out of touch with youth culture",
    "nostalgic for the good old days",
    "unsolicited advice-giver",
]

GEN_ALPHA_STYLES = [
    "NPC brainrot",
    "doomer and nihilistic",
    "ironic hype",
    "chronically online",
    "deadpan sarcastic",
    "extremely overdramatic",
    "low effort minimal energy",
    "delulu and delusional optimism",
    "rizz-coded smooth talker",
    "villain arc mentality",
    "main character vibes",
    "unhinged chaos energy",
]


# ──────────────────────────────────────────────
# 3. SENTENCE TYPE BUCKETS
# ──────────────────────────────────────────────
SENTENCE_TYPES = [
    "questions",
    "commands or instructions",
    "complaints or frustrations",
    "compliments or praise",
    "warnings or advice",
    "observations or statements",
    "exclamations or reactions",
    "requests or suggestions",
]


# ──────────────────────────────────────────────
# 4. SEED SENTENCE BANK  (~20 per topic)
# ──────────────────────────────────────────────
SEED_BANK: dict[str, list[str]] = {
    "school and studying": [
        "You need to buckle down and focus on your academics.",
        "I spoke with your teacher and she says you're not applying yourself.",
        "Back in my day, we didn't have calculators and we turned out fine.",
        "If you don't get good grades, you won't get into a good college.",
        "You should be spending at least two hours studying every evening.",
        "Turn off that music; you cannot possibly concentrate with that noise.",
        "I don't understand why school is so hard for you when you're so smart.",
        "Have you started on that project yet? It's due next week.",
        "You need to read more books instead of staring at that screen.",
        "Participation in class is a sign of engagement with your education.",
        "Your GPA will follow you for the rest of your life.",
        "I'm going to email your professor about your grade.",
        "Study groups were very effective when I was in school.",
        "You should sit in the front row so you can pay better attention.",
        "This material will be on the exam, so you should take notes.",
        "I don't understand why you need a tutor; just study harder.",
        "When I was your age, I was working and going to school at the same time.",
        "You need to review your notes every single night, not just before the test.",
        "Extra credit opportunities are not something to be dismissed.",
        "Your future employers will look at your transcripts.",
    ],
    "food and cooking": [
        "You need to eat a proper breakfast before you leave the house.",
        "That fast food is going to ruin your health.",
        "I don't understand why you can't just eat what I cook.",
        "Have you tried adding a little salt? That's all it needs.",
        "Vegetables are very important for your development.",
        "You should learn to cook at your age; it's an essential life skill.",
        "We're having dinner as a family at six o'clock sharp.",
        "I made this from scratch. The least you could do is try it.",
        "Snacking between meals is why you're not hungry at dinner.",
        "This recipe has been in our family for three generations.",
        "You can't live on instant noodles forever, you know.",
        "Organic food is worth the extra cost for your health.",
        "Don't play with your food; there are children starving in the world.",
        "You need to drink eight glasses of water a day.",
        "I'm concerned you're not getting enough protein.",
        "Why would you order delivery when I can make the same thing at home?",
        "That coffee will stunt your growth.",
        "Leftovers are perfectly good food. There's no need to waste.",
        "You should sit down and eat properly, not in front of the television.",
        "Home-cooked meals are always healthier than restaurant food.",
    ],
    "sports and fitness": [
        "You need to get off that couch and get some exercise.",
        "Team sports build character and discipline.",
        "I don't understand why you won't at least try the gym.",
        "Fresh air and physical activity are good for the brain.",
        "You should stretch before and after exercise to avoid injury.",
        "In my day, we played outside until the street lights came on.",
        "Hydration is very important when you're doing physical activity.",
        "You need to push through the discomfort if you want to improve.",
        "Proper form is more important than how much weight you lift.",
        "You should join a sports team; it's good for social development.",
        "Walking is perfectly good exercise. You don't need a fancy gym.",
        "I'm concerned you're not getting enough physical activity.",
        "Rest days are just as important as workout days.",
        "You need a good pair of running shoes if you're going to run.",
        "Diet and exercise go hand in hand.",
        "The mental benefits of exercise are just as important as the physical ones.",
        "You shouldn't skip leg day.",
        "Consistency is more important than intensity when starting out.",
        "Professional athletes didn't get there without hard work and sacrifice.",
        "Your posture when you sit at that computer is going to cause back problems.",
    ],
    "relationships and dating": [
        "I just want to make sure this person is treating you with respect.",
        "You're too young to be so serious about someone.",
        "Have you met their parents yet? That's an important step.",
        "You should focus on yourself before getting into a relationship.",
        "Long-distance relationships never work out in the end.",
        "I just think you could do better, that's all I'm saying.",
        "You shouldn't be texting someone at all hours of the night.",
        "In my day, you actually called someone on the telephone to ask them out.",
        "You need to communicate openly and honestly in a relationship.",
        "I'm not sure that person has the best intentions.",
        "Don't let anyone pressure you into something you're not comfortable with.",
        "You spend too much time worrying about relationships at your age.",
        "Compromise is the foundation of any healthy relationship.",
        "You should never change who you are for someone else.",
        "I just hope you're being careful with your heart.",
        "That relationship doesn't sound very healthy to me.",
        "You need to put in the effort if you want the relationship to work.",
        "First impressions are very important; make sure you dress nicely.",
        "Trust takes a long time to build and only a moment to destroy.",
        "You'll understand when you're older why this isn't a good idea.",
    ],
    "technology and social media": [
        "I don't understand why you need to post everything online.",
        "That app is probably selling your personal information.",
        "You need to be careful about what you share on the internet.",
        "Can you come help me figure out why my computer is doing this?",
        "Social media is making your generation very unhappy.",
        "You're going to ruin your eyesight staring at that screen all day.",
        "I don't understand why you need a new phone when this one works perfectly fine.",
        "Back in my day, we looked things up in encyclopedias.",
        "You should be talking to people face to face, not through a screen.",
        "I think your phone is listening to our conversations.",
        "I don't understand why you need so many passwords.",
        "You should turn off your Wi-Fi at night for your health.",
        "These influencers are not good role models for young people.",
        "I'm worried about what strangers on the internet might say to you.",
        "Why do you need to charge your phone again? You just charged it.",
        "I read an article saying screen time is very bad for sleep.",
        "Can you please put your phone away when we're having dinner?",
        "You shouldn't believe everything you read on the internet.",
        "I don't understand why you'd want strangers to know your business.",
        "What happened to just picking up the phone and calling someone?",
    ],
    "money and work": [
        "You need to start saving now or you'll regret it later.",
        "Money doesn't grow on trees, you know.",
        "You should have at least three months of expenses in savings.",
        "I don't understand why you're spending so much on coffee every day.",
        "Hard work and dedication are how you get ahead in life.",
        "You should invest in a 401k as soon as you start working.",
        "That purchase was completely unnecessary.",
        "You need to establish a budget and stick to it.",
        "Starting salary is not the end-all. It's about growth potential.",
        "In my day, you stayed at a job for thirty years and got a pension.",
        "You should dress professionally even for a video interview.",
        "Credit card debt will follow you for years.",
        "Networking is very important for career advancement.",
        "You should negotiate your salary instead of just accepting the first offer.",
        "A college degree is the best investment you can make in yourself.",
        "Side hustles are a good way to supplement your income.",
        "You should keep your professional and personal life completely separate.",
        "I don't understand why you'd leave a perfectly good job.",
        "Your attitude toward your boss will determine how far you go.",
        "Financial literacy is something they really should teach in school.",
    ],
    "family and home": [
        "You need to call your grandparents more often.",
        "Family dinner is important and I'd like everyone to be present.",
        "Your room is a complete disaster. Please clean it up.",
        "I raised you better than that.",
        "You need to do your chores without being asked.",
        "When you have your own house, you can do things your way.",
        "Family comes first. That's just how it is.",
        "You should appreciate what you have.",
        "Your cousin just graduated with honors. Isn't that something?",
        "We didn't have the things you have growing up and we were grateful.",
        "The least you could do is help out around the house.",
        "I'm not your maid.",
        "You need to learn basic home maintenance skills.",
        "The holidays are important to this family and I expect you to be there.",
        "Your attitude lately has been very concerning to me.",
        "I just want what's best for you, even if it doesn't feel that way.",
        "You can't stay in bed all day. That's not healthy.",
        "You should answer your phone when family calls.",
        "We need to have a family meeting about some things.",
        "This house has rules and I expect them to be followed.",
    ],
    "travel and going out": [
        "Make sure you let me know when you arrive safely.",
        "I don't understand why you need to go out every weekend.",
        "You should research the area before you visit.",
        "Travel is very educational if you do it properly.",
        "I'm not sure that neighborhood is safe at night.",
        "You should always have a backup plan when you travel.",
        "Make sure you get travel insurance. It's worth it.",
        "You need to be home by a reasonable hour.",
        "I read that the food there is very different from what you're used to.",
        "You should dress appropriately for the climate.",
        "Always keep your passport in a safe place.",
        "I don't understand why you'd want to backpack instead of staying in a hotel.",
        "You should call me when you land. I'll be worried.",
        "Make sure you have enough local currency before you arrive.",
        "Going out every night is expensive and not sustainable.",
        "I'm concerned about your safety when you travel alone.",
        "You should take pictures so you remember the experience.",
        "The traffic in that city is absolutely terrible.",
        "You should try the local cuisine; that's the best part of traveling.",
        "Make sure you have my number memorized in case your phone dies.",
    ],
    "mental health and feelings": [
        "You just need to push through it. That's what we did.",
        "I don't understand why your generation is so anxious about everything.",
        "Have you tried going for a walk when you're feeling down?",
        "You need to stop overthinking and just deal with it.",
        "Talking to a professional is nothing to be ashamed of.",
        "I'm worried about you. You don't seem like yourself lately.",
        "You need to learn to manage your stress in healthy ways.",
        "Back in my day, we didn't have time to be depressed.",
        "You should try journaling. It might help you process your feelings.",
        "I think you spend too much time alone in your room.",
        "It's okay to not be okay sometimes, but you can't stay there.",
        "Have you tried meditating? I read it's very good for anxiety.",
        "You can't let other people's opinions affect you so much.",
        "A positive attitude makes a huge difference.",
        "You need to get enough sleep. It affects everything.",
        "I'm here if you want to talk. I just don't always know what to say.",
        "Your feelings are valid but you still have responsibilities.",
        "You put too much pressure on yourself.",
        "Try to focus on what you can control and let go of the rest.",
        "I love you even when I don't understand what you're going through.",
    ],
    "pop culture and entertainment": [
        "I don't understand why that person is famous.",
        "The music from your generation has no real melody.",
        "We didn't need all this streaming; we just watched what was on.",
        "That movie has too much violence and language.",
        "I don't understand why you'd pay to watch someone play video games.",
        "The classics are classics for a reason.",
        "Back in my day, you had to wait a week for the next episode.",
        "I don't understand the appeal of reality television.",
        "That comedian isn't funny; they just say offensive things.",
        "Books are always better than the movie adaptation.",
        "I don't understand why they keep remaking perfectly good films.",
        "The lyrics to this song don't even make sense.",
        "We used to go to the video store to rent movies on a Friday night.",
        "That show has a very interesting plot, I must admit.",
        "I just don't understand why everything has to be so dark and violent.",
        "That celebrity is not a good role model for young people.",
        "The special effects back then were more creative because they had limitations.",
        "You should watch some classic films to understand cinema.",
        "I'm not sure that content is appropriate for someone your age.",
        "Concerts cost an absolutely outrageous amount of money these days.",
    ],
    "health and medicine": [
        "You should see a doctor about that. Don't put it off.",
        "I read that that medication has very serious side effects.",
        "You need to take better care of yourself.",
        "Make sure you're getting enough vitamins.",
        "Sleep is when your body does its healing.",
        "You should wash your hands more often.",
        "I don't understand why you'd trust the internet over a real doctor.",
        "You need to get a full checkup every year.",
        "That sounds like it could be something serious. Please get it checked.",
        "You shouldn't be self-diagnosing from the internet.",
        "Sunscreen is non-negotiable. Skin cancer is real.",
        "Your immune system needs proper nutrition to function.",
        "I've been taking this supplement for years and it really works.",
        "You need to floss, not just brush.",
        "Mental health is just as important as physical health.",
        "That diet sounds very extreme. Please talk to a nutritionist.",
        "You should be taking a multivitamin every morning.",
        "Don't ignore symptoms. They're your body trying to tell you something.",
        "I think you might be getting sick. You look pale.",
        "Rest is the best medicine when you're not feeling well.",
    ],
    "politics and news": [
        "You need to stay informed about what's happening in the world.",
        "I don't understand why young people don't vote.",
        "Things were different when I was growing up.",
        "The news nowadays is completely biased.",
        "You need to read multiple sources to get a complete picture.",
        "In my day, you didn't discuss politics or religion at the dinner table.",
        "I'm concerned about the direction this country is going.",
        "You shouldn't believe everything the media tells you.",
        "Your generation is going to inherit these problems.",
        "Politicians are the same no matter which party they belong to.",
        "Civic duty is something that everyone should take seriously.",
        "I don't understand why everything has to become political.",
        "You should form your own opinions and not just follow the crowd.",
        "The economy was much simpler when I was young.",
        "You need to think about the long-term consequences of these policies.",
        "Social media has made political discourse completely uncivil.",
        "You should respect the democratic process even when you disagree.",
        "Things are more complicated than they appear on the surface.",
        "History tends to repeat itself, which is why you need to study it.",
        "You should write to your elected representative if you feel strongly.",
    ],
    "fashion and style": [
        "You're not leaving the house dressed like that.",
        "I don't understand why you'd pay so much for ripped jeans.",
        "You should dress for the job you want, not the job you have.",
        "In my day, people took pride in their appearance.",
        "Those shoes are going to destroy your feet.",
        "You look very nice when you make an effort.",
        "I don't understand why you have to follow every trend.",
        "First impressions matter more than people admit.",
        "That color doesn't suit you at all.",
        "You could look so presentable if you just tried a little.",
        "I don't understand why you'd want to dress like everyone else.",
        "Those tattoos are going to affect your job prospects.",
        "You should invest in a few quality pieces rather than lots of cheap ones.",
        "I don't understand the appeal of wearing your pants like that.",
        "You dressed so nicely when you were younger.",
        "That hairstyle is very... expressive.",
        "Classic style never goes out of fashion.",
        "You should dress appropriately for the occasion.",
        "I just want you to look put-together when we go out.",
        "That perfume is very strong. Is that intentional?",
    ],
    "gaming and hobbies": [
        "You've been playing that game for four hours straight.",
        "I don't understand how that's a real career.",
        "You should find a hobby that gets you off the couch.",
        "That game seems very violent. Is it appropriate?",
        "Back in my day, we played board games as a family.",
        "You're wasting your potential sitting in front of that screen.",
        "I read that video games are linked to aggressive behavior.",
        "You should develop a hobby you can do for the rest of your life.",
        "I don't understand why people watch other people play games.",
        "You could be learning a musical instrument with that time.",
        "Everything in moderation, including video games.",
        "I just don't understand the appeal.",
        "That gaming equipment cost how much?",
        "You should take regular breaks from the screen for your eyes.",
        "Hobbies are important for your mental well-being.",
        "You were so creative as a child. I miss that.",
        "I'm glad you have something you're passionate about, at least.",
        "Can't you find something more productive to do with your time?",
        "I worry about how much time you spend doing this.",
        "You should have a backup plan in case the gaming thing doesn't work out.",
    ],
    "weather and environment": [
        "You need to take an umbrella. It looks like rain.",
        "The weather these days is not what it used to be.",
        "I'm concerned about what we're doing to this planet.",
        "You should recycle. It's the responsible thing to do.",
        "In my day, we didn't have air conditioning and we survived.",
        "You should dress in layers. The weather can be unpredictable.",
        "I don't understand why it's so warm for this time of year.",
        "Make sure you're not wasting electricity.",
        "The ozone layer is something your generation needs to take seriously.",
        "It's too cold to go out without a proper coat.",
        "We need to conserve water. It's a precious resource.",
        "The storms lately have been very severe.",
        "I'm not sure what the weather is going to do today.",
        "You should appreciate nature more instead of being inside all day.",
        "Climate change is a very complicated issue.",
        "Make sure you have enough antifreeze in the car for winter.",
        "The pollen count is very high. Take your allergy medication.",
        "You should turn off lights when you leave the room.",
        "We should plant some trees. It's good for the environment.",
        "The heat index makes it feel much hotter than it actually is.",
    ],
}


# ──────────────────────────────────────────────
# 4. ROTATING FEW-SHOT POOL  (30 examples)
# ──────────────────────────────────────────────
FEW_SHOT_POOL = [
    ("I don't understand why you spend so much time on your phone.", "bro is literally glued to his phone rn no cap"),
    ("You need to apply yourself more in school.", "bestie just lock in already it's not that deep"),
    ("Back in my day, we didn't need all these gadgets.", "ok grandpa we get it you walked uphill both ways"),
    ("I'm very concerned about your future prospects.", "ngl your future era is giving struggle bus"),
    ("Could you please turn down that music?", "the music is literally at volume 2 chill out"),
    ("You should be saving money instead of spending it.", "me and my wallet are in our flop era"),
    ("You have so much potential if you would only apply yourself.", "you could literally be that girl but you choose chaos"),
    ("That young man seems like a bad influence on you.", "he's giving villain arc and honestly i'm here for it"),
    ("Why don't you go outside and get some fresh air?", "skill issue if you think im touching grass today"),
    ("You're always on that computer. It can't be good for you.", "me and my screen time are in a whole situationship"),
    ("You need to be more responsible with your choices.", "the audacity to make decisions and face consequences"),
    ("Make sure you eat a proper breakfast before school.", "breakfast hits different when it's 2pm ngl"),
    ("You should call your grandmother more often.", "grandma understood the assignment she's valid fr"),
    ("I'm not paying for that nonsense.", "the way my wallet said nope and left the chat"),
    ("You need to get a good night's sleep.", "me at 3am: one more video. me at 7am: i am deceased"),
    ("That's not how real life works.", "the real world said no cap and i said say less"),
    ("I just want what's best for you.", "the way you're lowkey trying to give unsolicited advice again"),
    ("You need to learn to manage your time better.", "my schedule said it's giving dumpster fire era"),
    ("Why can't you just be normal?", "normal is not in my vocabulary bestie"),
    ("You're throwing your life away.", "im not throwing it im yeeting it strategically"),
    ("I'm worried about the people you're spending time with.", "my friend group understood the assignment they're built different"),
    ("You should dress more appropriately.", "my fit is giving main character and you just can't see it"),
    ("Money doesn't grow on trees.", "sir the economy said trees would be more reliable"),
    ("You need a backup plan.", "my backup plan's backup plan has a backup plan"),
    ("I don't know what's gotten into you lately.", "i had a villain arc it was necessary for my growth"),
    ("That's completely irresponsible.", "i prefer the term chaotically spontaneous"),
    ("You could achieve anything if you tried harder.", "i'm locked in i'm just locked into the wrong things"),
    ("We need to talk about your attitude.", "the vibe check failed and now there's consequences"),
    ("You should be more grateful.", "i'm grateful i'm just also tired and down bad"),
    ("Life is not a game, you know.", "actually it's a roguelike and i just lost all my saves"),
]


# ──────────────────────────────────────────────
# 5. FUZZY DEDUPLICATION
# ──────────────────────────────────────────────
_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "must","shall","to","of","in","for","on","with","at","by","from","up",
    "about","into","through","i","you","he","she","it","we","they","me",
    "him","her","us","them","my","your","his","its","our","their","this",
    "that","these","those","and","but","or","nor","so","yet","not","no",
    "very","just","more","most","such","than","too","also","any","all",
    "each","every","both","few","other","some","only","own","same","need",
    "want","get","go","know","think","see","come","take","make","say",
}

def normalize(text: str) -> str:
    """Lowercase -> strip punctuation -> remove stopwords -> sort -> join."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = sorted(w for w in text.split() if w not in _STOPWORDS and len(w) > 2)
    return " ".join(words)


# ──────────────────────────────────────────────
# PROMPT BUILDER
# ──────────────────────────────────────────────
def build_prompt(
    topic: str,
    boomer_style: str,
    gen_alpha_style: str,
    sentence_type: str,
    few_shots: list[dict],
    seeds: list[str] | None,
    n: int,
) -> str:

    few_shot_block = "\n".join(
        f'  Boomer: "{p["boomer"]}"\n  Gen Alpha: "{p["gen_alpha"]}"'
        for p in few_shots
    )

    seed_block = ""
    if seeds:
        seed_list = "\n".join(f"  - {s}" for s in seeds)
        seed_block = (
            f"\nSeed sentences (use as inspiration or translate — "
            f"do NOT copy verbatim, generate NEW sentences):\n{seed_list}\n"
        )

    return f"""You are a linguist specializing in generational language translation.

Batch parameters:
  Topic:            {topic}
  Boomer style:     {boomer_style}
  Gen Alpha style:  {gen_alpha_style}
  Sentence type:    {sentence_type}

Reference examples (do NOT reproduce these):
{few_shot_block}
{seed_block}
Your task:
Generate exactly {n} NEW phrase pairs translating {boomer_style} boomer sentences
into {gen_alpha_style} Gen Alpha slang.

Hard rules:
- Sentence type focus: {sentence_type}
- Topic: sentences must relate to {topic}
- Do NOT reproduce any example or seed sentence verbatim
- Do NOT use emojis
- Vary sentence length: short (1 clause), medium (2 clauses), long (3+)
- Output ONLY a raw JSON array, no markdown fences, no explanation

Format: [{{"boomer": "...", "gen_alpha": "..."}}, ...]

Generate {n} pairs now:"""


# ──────────────────────────────────────────────
# API CALL WITH RETRY + EXPONENTIAL BACKOFF
# ──────────────────────────────────────────────
def generate_batch(
    topic: str,
    boomer_style: str,
    gen_alpha_style: str,
    sentence_type: str,
    few_shots: list[dict],
    seeds: list[str] | None,
    n: int,
    temperature: float,
) -> list[dict]:

    prompt = build_prompt(
        topic, boomer_style, gen_alpha_style, sentence_type, few_shots, seeds, n
    )
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You output only valid JSON arrays. No markdown, no preamble."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                timeout=60,
            )
            text = response.choices[0].message.content.strip()

            # Strip markdown fences if model misbehaves
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            pairs = json.loads(text)

            valid = []
            for p in pairs:
                if isinstance(p, dict) and "boomer" in p and "gen_alpha" in p:
                    b = str(p["boomer"]).strip()
                    g = str(p["gen_alpha"]).strip()
                    if b and g:
                        valid.append({
                            "boomer":          b,
                            "gen_alpha":       g,
                            "topic":           topic,
                            "boomer_style":    boomer_style,
                            "gen_alpha_style": gen_alpha_style,
                            "sentence_type":   sentence_type,
                        })
            return valid

        except openai.RateLimitError:
            wait = 2.0 * (2 ** attempt) + random.uniform(0, 1)
            print(f"\n  [rate limit] sleeping {wait:.1f}s ...")
            time.sleep(wait)

        except openai.APIError as e:
            print(f"\n  [API error] attempt {attempt+1}: {e}")
            time.sleep(2.0 * (attempt + 1))

        except json.JSONDecodeError as e:
            print(f"\n  [JSON parse error] attempt {attempt+1}: {e}")
            time.sleep(1)

    return []


# ──────────────────────────────────────────────
# APPEND TO CSV (incremental — crash-safe)
# ──────────────────────────────────────────────
def append_to_csv(rows: list[dict]) -> None:
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerows(rows)


# ──────────────────────────────────────────────
# LOAD EXISTING PROGRESS
# ──────────────────────────────────────────────
dataset_rows: list[dict] = []
exact_seen:   set[str]   = set()   # exact boomer strings
fuzzy_seen:   set[str]   = set()   # normalized boomer strings

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_rows.append(row)
            exact_seen.add(row["boomer"])
            fuzzy_seen.add(normalize(row["boomer"]))
    print(f"[resume] Loaded {len(dataset_rows):,} existing pairs.")
else:
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    print(f"[init] Created {OUTPUT_FILE}")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
already_have = len(dataset_rows)
needed       = TOTAL_TARGET - already_have

print(f"\nTarget: {TOTAL_TARGET:,} | Have: {already_have:,} | Need: {needed:,}")
print(f"Model: {MODEL} | Temps: {TEMPERATURES}")
print(f"Style combos: {len(BOOMER_STYLES) * len(GEN_ALPHA_STYLES):,} possible\n")

if needed <= 0:
    print("Already at target. Done!")
else:
    few_shot_cursor = 0

    with tqdm(total=TOTAL_TARGET, initial=already_have, unit="pair", desc="Generating") as pbar:
        while len(dataset_rows) < TOTAL_TARGET:

            # Sample batch parameters randomly
            topic          = random.choice(TOPIC_BUCKETS)
            boomer_style   = random.choice(BOOMER_STYLES)
            gen_alpha_style = random.choice(GEN_ALPHA_STYLES)
            sentence_type  = random.choice(SENTENCE_TYPES)
            temperature    = random.choice(TEMPERATURES)

            # Rotate 3 few-shot examples from pool
            pool_size  = len(FEW_SHOT_POOL)
            few_shots  = [
                {"boomer": FEW_SHOT_POOL[(few_shot_cursor + i) % pool_size][0],
                 "gen_alpha": FEW_SHOT_POOL[(few_shot_cursor + i) % pool_size][1]}
                for i in range(3)
            ]
            few_shot_cursor = (few_shot_cursor + 3) % pool_size

            # Seed mode vs pure generation mode
            seeds = None
            if random.random() < SEED_BATCH_PROB and topic in SEED_BANK:
                seed_pool = SEED_BANK[topic]
                seeds = random.sample(seed_pool, min(8, len(seed_pool)))

            # Call API
            remaining  = TOTAL_TARGET - len(dataset_rows)
            batch_size = min(PAIRS_PER_CALL, remaining)

            raw_pairs = generate_batch(
                topic           = topic,
                boomer_style    = boomer_style,
                gen_alpha_style = gen_alpha_style,
                sentence_type   = sentence_type,
                few_shots       = few_shots,
                seeds           = seeds,
                n               = batch_size,
                temperature     = temperature,
            )

            # Fuzzy deduplicate
            unique = []
            for p in raw_pairs:
                norm = normalize(p["boomer"])
                if p["boomer"] not in exact_seen and norm not in fuzzy_seen:
                    exact_seen.add(p["boomer"])
                    fuzzy_seen.add(norm)
                    dataset_rows.append(p)
                    unique.append(p)

            if unique:
                append_to_csv(unique)

            pbar.update(len(unique))
            time.sleep(0.5)   # stay under 500 RPM rate limit

    print(f"\nDone! {len(dataset_rows):,} pairs saved to {OUTPUT_FILE}")
