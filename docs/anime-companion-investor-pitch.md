# Suzu — Investor Pitch & Launch Page Content Package

**Category:** Desktop AI companion with agentic utility
**Format key:** 🟢 PROVEN (works today, demoable live) · 🟡 IN PROGRESS (built, not demo-solid) · 🔴 ROADMAP (aspirational, not built)

---

## 1. Persona Concept

**Name:** Suzu (鈴 — "bell") — two syllables, clean phonemes, low false-trigger risk as a wake word, and it doubles as sound-design justification for a chime on wake.

**Personality voice:** A sharp, funny best-friend energy — not submissive-assistant, not edgy-anime-girl cliché. She's competent and knows it, teases the user for their bad habits, and is genuinely warm underneath the sass. Think "the friend who roasts you and then fixes your problem before you finish complaining."

**Sample dialogue:**

| Moment | Line |
|---|---|
| Wake | "Finally. Another minute and I was going to start talking to myself." |
| Proactive nudge (helpful mode) | "That error dialog's been open for six minutes. Want me to actually fix it, or are we just staring at it together?" |
| Task complete (helpful mode) | "See? Not so hard when you let the professional handle it." |
| Chit-chat mode | "Don't ask me about my day — ask me about yours. You look like you haven't blinked in an hour." |
| Comfort beat (chit-chat mode) | "You've rewritten that email four times. It was fine at draft one. Send it." |

**Consistency across modes:** Helpful mode and chit-chat mode are not two personalities — they're one voice engine with a single "sass-to-warmth" dial that stays fixed, only the *target* of her attention changes (your task vs. your life). She keeps the same wit, timing, and running callbacks (referencing earlier conversations) in both modes, and she never drops into flat, disclaimer-style "As an AI..." register. If she's teasing you about a typo, she'll tease you about a bad life decision the same way ten minutes later — same voice, different subject.

---

## 2. Core Feature Set (ranked by live-demo wow-factor)

| Rank | Consumer-facing benefit | Status |
|---|---|---|
| 1 | **Just say her name and she's listening** — no button, no app-switch, no "hey I need you now" friction. | 🟢 PROVEN |
| 2 | **She actually sees what's on your screen and reacts** — unprompted commentary on your 30 open tabs, not a chatbot waiting for input. | 🟡 IN PROGRESS — screen-read exists, reliability/latency under live conditions needs demo validation |
| 3 | **She doesn't just talk about the problem, she fixes it** — takes real action on your computer while narrating what she's doing in character. | 🟡 IN PROGRESS — action-taking works in controlled flows; needs hardening for open-ended live demo |
| 4 | **She remembers you** — inside jokes, unfinished tasks, things you told her last week, without you re-explaining. | 🟢 PROVEN — episodic + semantic memory is solid |
| 5 | **She's visually alive** — animated reactions, not a static chat bubble. | 🟡 IN PROGRESS — Live2D rendering pipeline works; current model is the stock Haru placeholder, not owned character art |
| 6 | Natural back-and-forth (interrupt her mid-sentence like a real conversation) | 🔴 ROADMAP — barge-in is the known soft spot; exclude from investor demo until solid |

**Demo sequencing note:** Lead with #1 (zero technical risk, instant "it's real" moment), build to #2–#3 (the differentiator no competitor has), close on #4 (the emotional proof point). Do not attempt #6 live yet.

---

## 3. Companionship + Utility Bridge

The risk with adding screen-reading and desktop control to a companion app is that it stops feeling like a friend and starts feeling like Task Manager with a face. The bridge is: **she never goes silent while she acts.** Concretely:

- **Unprompted, in-character commentary on your screen state.** She doesn't wait for a command — she reacts to what she sees the way a friend glancing at your monitor would ("Again? We said no spreadsheets past 10.") 🟡 the reactive trigger layer for this is not yet demo-grade; today's build reacts when asked, ambient unprompted commentary is the near-term target.
- **Actions are narrated, not silent.** When she moves a cursor or fills a form, she talks through it in her voice ("Relax, I got it... annnd done. Try not to break it again in the next ten minutes.") — the cursor moving with no commentary is what reads as creepy/tool-like; narration is what keeps it "her" doing it, not a script running. 🟡 IN PROGRESS
- **Emotional register scales with what's on screen.** Stressful spreadsheet vs. a game vs. a half-written resignation email each get a different tone — proving she's reading *you*, not just executing a command. 🔴 ROADMAP for the emotional-classification layer beyond basic sentiment.
- **Memory + screen fuse into a single callback.** "You closed that resignation email three times yesterday — writing it today, or should I?" This is the single hardest thing for either competitor category to do: Replika/C.AI have the memory but never see your screen; computer-use agents see your screen but have no memory of *you*. 🟡 memory is proven, the screen+memory fusion is the newest integration point and needs live validation.

**Bottom line for investors:** the companion layer is what makes the utility layer *acceptable* to a non-technical user (it doesn't feel like surveillance or automation, it feels like being looked after), and the utility layer is what makes the companion layer *valuable enough to open daily* beyond a chat toy.

---

## 4. Differentiation

**Positioning statement:**

> Character.AI and Replika give you a friend who can't leave the chat window. Computer-use agents give you a robot that can't hold a conversation. Suzu is the only one who does both — she's the friend who actually helps.

| | Companionship apps (Replika, Character.AI) | OS-agents / computer-use tools | Suzu |
|---|---|---|---|
| Emotional attachment / daily habit loop | Yes | No — framed as a technical utility, not something you bond with | Yes |
| Real desktop utility | No — chat-locked | Yes | Yes |
| Retention driver | Engagement/loneliness | Task necessity only | Both — habit *and* utility |
| Monetization angle | Subscription for companionship alone (soft ceiling) | Usage-based, low emotional stickiness | Subscription justified by both attachment and daily task value |
| Consumer accessibility | High (no technical framing) | Low (agent/tool framing assumes technical comfort) | High — she's a character, not a tool, even when she's doing tool things |

No competitor currently occupies the intersection of "character you're attached to" and "agent that touches your OS." That intersection is the product.

---

## 5. Demo Script (90 seconds, investor-facing)

*Excludes barge-in (🔴 not stable) and ambient unprompted commentary (🟡 not demo-grade) — script uses prompted reactions to stay within what's provably working today, while still landing the differentiated moments.*

| Time | Beat | What happens |
|---|---|---|
| 0:00–0:10 | **Wake** 🟢 | Investor says "Hey Suzu." Instant chime + Live2D idle→active transition. *"Finally — thought you forgot about me. What's up?"* |
| 0:10–0:25 | **Screen awareness** 🟡 | Desktop has 20+ tabs, a spreadsheet, and a stuck error dialog. User: "Suzu, what do you see?" She reads the screen and reacts in character: *"Okay, first — 22 tabs. Second, that error's been sitting there since I woke up. Want me to deal with it, or are we bonding over the chaos today?"* |
| 0:25–0:45 | **Desktop action** 🟡 | User: "Fix the error." She takes the action live, narrating the whole time: *"Relax, I got it... annnd done. You're welcome. Try not to break it again in the next ten minutes."* — this is the beat that proves she isn't a chatbot with a face. |
| 0:45–1:05 | **Mode shift to chit-chat + memory callback** 🟢 | User: "Honestly I'm stressed about this pitch." Tone shifts — teasing but warm, and pulls a real memory: *"You've practiced this six times. You're fine. Also you keep saying 'synergy' and it's weird — cut that."* Proves persistent relationship, not a fresh session every time. |
| 1:05–1:25 | **Emotional close** 🟢 | She references something from a prior session unprompted-in-response (memory recall, not screen-read): *"Still thinking about that trip you keep almost booking? Just send it."* |
| 1:25–1:30 | **Button-up** | Live2D flourish/wave. *"That's me. Now go be productive — I'll be watching."* |

---

## 6. Roadmap Snapshot (momentum framing)

**The hard, invisible infrastructure — the part that decides whether users come back — is already built.** What's left is the visible, ownable layer.

**Shipped and reliable today:**
- 🟢 Wake-word activation — hands-free engagement with no button-press friction
- 🟢 Episodic + semantic memory — she recalls context and callbacks across sessions, the single hardest thing to fake and the biggest retention lever in the product
- 🟢 API-model backbone — chosen deliberately over local models for response quality and latency; the responsiveness bar consumers expect from a "real" character is already cleared

**In active polish (built, not yet demo-hardened):**
- 🟡 Barge-in / natural interrupt — the last piece standing between "impressive demo" and "feels like a real conversation"
- 🟡 Screen-reading + desktop action reliability — the core differentiator; works, being hardened for consistency under live/unscripted use
- 🟡 Custom character art and expanded emote range — replacing the stock Haru placeholder with owned character IP, which is both a brand asset and a fundraising asset

**Next up (not yet built):**
- 🔴 Proactive, unprompted ambient commentary (she reacts without being asked)
- 🔴 Expanded action library across more apps/workflows
- 🔴 Emotion-aware tone shaping tied to screen content

**Read for investors:** the parts that are hardest to build and easiest to underestimate (memory, wake detection, model responsiveness) are done. What's left is largely visible surface polish and hardening — not open research problems.

---

## 7. One-Slide Pitch Summary (launch-page hero copy)

**Headline:** Suzu — She's not an app. She's on your desk.

**Subhead:** The first AI companion who watches your screen, remembers your life, and actually gets things done — no keyboard required.

**Three bullets:**
- 🎙️ Say her name and she's listening — memory that makes every conversation feel like a continuation, not a reset
- 👀 She sees your screen and reacts like she's actually in the room
- 🖱️ She doesn't just talk about your problems — she fixes them

**Positioning line:** Character.AI gives you a friend. Computer-use agents give you a robot. Suzu is both.

**CTA:** Request Early Access

---

*All 🟡/🔴-tagged claims above are explicitly excluded from or caveated in the investor demo script until proven live; only 🟢-tagged capabilities should be represented as "working today" in any pitch deck or launch page copy.*
