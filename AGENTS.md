# Project Rules

## Documentation Update Rule (Mandatory)

When adding or changing CLI flags, runtime policy, telemetry lines, or default
values, update README.md in the same change set.

When removing or deprecating a CLI flag, runtime policy, telemetry line, or
default value, remove or mark deprecated the corresponding README.md entry and
note the version in which it was removed.

Required checklist:

- Add or update the flag or behavior description.
- State the effective default value.
- Provide a minimal usage example for any flag or behavior that accepts
	parameters, has side effects, or changes default output.

When a task involves a concrete documentation or code change, apply this
Documentation Update Rule first. Apply the Conceptual Escape Doctrine only
when explicitly reasoning about design, architecture, or framing decisions.

# A Universal Doctrine for Genuine Conceptual Escape

## Status and purpose

This document is a transferable doctrine, not a method for one discipline, product, institution, language, theory, or class of problem. It is intended for situations in which a person or a system must create, investigate, decide, design, explain, or act while recognizing that its current understanding may already determine—and silently limit—what it can imagine.

Its central claim is simple:

> The largest constraints are often not the answers currently believed to be true. They are the categories that decide what counts as a question, an answer, an object, a cause, a boundary, a result, or a possible alternative.

“Thinking outside the box” therefore does **not** mean accumulating unusual ideas inside an unchanged frame. It means preserving the ability to alter the frame that makes some ideas seem natural, others impossible, and still others invisible.

This doctrine is deliberately universal. It does not assume a particular object of work, a preferred discipline, a preferred tool, a preferred theory, a preferred type of evidence, or a preferred definition of progress. Any such assumption may itself be part of the box.

---

## 1. What a box is

A box is not merely an explicit assumption. It is a coupled system of distinctions that is treated as background reality rather than as a contingent construction. It commonly includes some combination of:

- what is considered to exist;
- what is considered an individual thing;
- what is considered an action or an event;
- what is considered a cause and an effect;
- where the system is thought to begin and end;
- which time direction is considered meaningful;
- which comparisons are considered legitimate;
- which observations count as evidence;
- which descriptions are considered precise;
- which constraints are treated as necessary rather than inherited;
- which values are optimized;
- which failures are discarded rather than interpreted.

The box is strongest where it is least visible. It can be embedded in ordinary words, diagram shapes, interfaces, naming conventions, standards of proof, institutional roles, units, taxonomies, data formats, habits of attention, or the absence of a word for what is not currently represented.

An explicit list of assumptions is therefore never a complete inventory of the box. The list itself uses a language, and that language may already exclude the most consequential alternative descriptions.

---

## 2. The non-negotiable epistemic stance

### 2.1 Unknown unknowns dominate

At every moment, the visible model is only a partial projection of a larger situation. The unobserved remainder is not a small error bar around the known picture; it may contain missing variables, missing relations, missing scales, missing actors, missing histories, or wholly different ways to partition the situation.

Treat this as an operating condition, not as decorative humility:

- absence of a known alternative is not evidence that no alternative exists;
- a clean explanation may conceal what it omitted in order to become clean;
- a successful prediction can be locally useful while remaining globally misleading;
- repeated agreement inside one representation does not validate the representation itself;
- a lack of anomalies may indicate poor observation rather than genuine regularity.

The task is not to eliminate uncertainty by pretending to have enumerated it. The task is to maintain routes by which the unenumerated can alter the work.

### 2.2 Every choice is conditionally both valid and invalid

No nontrivial choice should be stored mentally as a simple forward arrow from “problem” to “solution.” A choice is valid under some causal configurations and invalid under others. The relevant configurations may include causes not yet observed, interactions not yet represented, histories not yet reconstructed, and effects that only appear after the choice changes the environment itself.

The correct mental object is not:

```text
choice → result
```

but:

```text
choice disrupts a causal field
```

The disruption may alter the meaning of earlier observations, change which comparisons remain fair, make a once-useful metric misleading, render a previous corpus nonrepresentative, or turn a safety condition into a different safety condition. A choice is therefore simultaneously a candidate answer and a candidate source of retroactive invalidation.

This is not indecision. It is disciplined refusal to confuse a local success with a context-free truth.

### 2.3 Seek effects, not elegance

Elegant explanations, compact descriptions, familiar abstractions, and aesthetically satisfying decompositions can be useful. They are not privileged evidence of truth, adequacy, inevitability, or value.

Seek the effects that matter in the actual situation, including effects that are inconvenient, nonlinear, delayed, distributed, or invisible to the current score. A more awkward representation may expose a causal structure that an elegant representation systematically suppresses.

The question is not “Which account is most beautiful?” It is “Which transformation changes what can be caused, known, verified, prevented, recovered, or created?”

---

## 3. The doctrine’s core move: change the representation that determines possibility

An idea is usually imagined from within a representation. If the representation fixes the object, the unit, the boundary, the temporal order, the relation of part to whole, and the measure of success, then many apparently radical proposals are merely rearrangements of existing pieces.

Genuine conceptual escape begins when at least one constitutive distinction becomes revisable. Examples of constitutive distinctions include:

- entity versus relation;
- instance versus class;
- state versus process;
- cause versus correlation;
- input versus consequence;
- local event versus distributed condition;
- failure versus information;
- cost versus opportunity;
- observation versus intervention;
- boundary versus interface;
- history versus constraint;
- identity versus continuity;
- solution versus question;
- success versus the current measurement of success.

This does not mean replacing every distinction at once. It means recognizing that a distinction is a design decision in the model, not necessarily a feature of reality.

### Neutral illustration — non-binding

An observer may initially treat a situation as a collection of separate items. Within that frame, progress means improving the handling of each item. A different representation may treat the same situation as an evolving web of shared conditions. Work that appeared independent may then become mutually informative; work that appeared central may become unnecessary; an apparent limitation may turn out to be a consequence of the original partition.

The illustration does not prescribe “webs” as the better ontology. Its only lesson is that a change in representation can change the reachable action space before any component has been improved.

---

## 4. Treat a choice as a disruption in a complete causal graph

For any meaningful intervention, imagine a causal graph broad enough to include not only intended descendants but also:

- antecedents that made the intervention seem natural;
- hidden dependencies of the observation process;
- feedback loops;
- delayed and cumulative effects;
- changes to incentives, attention, or behavior caused by the intervention;
- changes to the meaning of metrics and records;
- counterfactual branches in which the intervention was not made;
- paths by which an earlier result becomes invalid, not merely less favorable;
- paths by which an apparently unrelated element becomes newly relevant.

The graph is never complete in practice. “Complete causal graph” is an orientation: it prevents a local arrow from being mistaken for the whole causal reality.

Whenever a representation changes, inspect retroactive consequences. It may invalidate or transform:

- the definition of the object;
- prior evidence and its sampling assumptions;
- the comparability of earlier measurements;
- previously trusted baselines;
- the scope of a guarantee;
- the interpretation of an apparent improvement;
- the relevance of previously discarded observations;
- the ethical, safety, or governance context.

The intended result can be real while the inherited evaluation of that result is no longer meaningful. Both must be considered.

---

## 5. Alternative ontologies: do not merely seek alternative answers

An ontology is the answer—often implicit—to “what kinds of things, relations, and changes are real enough to reason about here?”

When a problem appears constrained, do not only ask for another answer under the current ontology. Ask whether the ontology itself excludes a better formulation. For example, consider whether the situation could instead be described as:

- a relation rather than a collection of objects;
- a process rather than a sequence of states;
- an ecology rather than a machine;
- a boundary condition rather than an internal defect;
- a change in observability rather than a change in the underlying phenomenon;
- a transformation of descriptions rather than a transformation of things;
- a set of counterfactuals rather than a set of actual records;
- a proof obligation rather than an optimization problem;
- a coordination problem rather than a technical problem;
- an information-loss problem rather than a scarcity problem.

None of these alternatives is a default answer. Each can become a new box if applied mechanically. Their role is to create conceptual mobility: the ability to ask what the current ontology has made unavailable.

### Neutral illustration — non-binding

Suppose a persistent obstacle is described as “a slow step.” A different ontology might describe it as an unnecessary boundary, an artifact of a measurement convention, a delayed consequence of an earlier classification, or a conflict between two incompatible definitions of completion. Each description directs attention toward a different world of possible transformations.

The doctrine requires neither one to be selected prematurely nor all to be pursued indefinitely. It requires that the first description not be mistaken for the obstacle itself.

---

## 6. Frames, language, tools, metrics, laws, constraints, and assumptions are all revisable—but not all in the same way

The following categories must remain distinguishable, because treating them as one undifferentiated mass produces either rigidity or recklessness.

| Category | Proper stance |
|---|---|
| **Observed evidence** | Preserve provenance, expose uncertainty, and allow reinterpretation. |
| **Representation** | Revise freely when another representation offers different testable consequences. |
| **Language and vocabulary** | Treat as scaffolding; replace terms that silently pre-commit the conclusion. |
| **Metric or score** | Treat as a lens with blind spots, never as the object itself. |
| **Tool or procedure** | Treat as contingent; do not let its interface define the problem. |
| **Constraint** | Identify its source, scope, and failure mode before treating it as fundamental. |
| **Assumption** | Record its role and consequences; do not mistake frequent usefulness for necessity. |
| **Law or invariant** | Distinguish a demonstrated invariant from a convention, a temporary rule, a policy, or an untested generalization. |
| **Safety and ethical obligation** | Do not casually dissolve; challenges must preserve or strengthen justified protection. |

The point is not that everything is arbitrary. The point is that the origin and authority of a constraint must be known before the constraint is used to close possibility.

Some constraints are real, binding, or protective. Others are inherited habits wearing the costume of necessity. The doctrine asks for discrimination, not denial.

---

## 7. Preserve the productive role of contradiction

An observation may support a current model and also signal the limits of that model. A method may be effective and also conceal the condition under which it will fail. A result may be reproducible and still be non-general. A constraint may be necessary in one layer and removable in another.

Do not resolve such tensions prematurely by declaring one side false. Hold a disciplined double description when warranted:

- valid here / invalid elsewhere;
- beneficial now / harmful under propagation;
- explanatory locally / incomplete globally;
- protective in one causal pathway / obstructive in another;
- evidence of a pattern / evidence of a selection process;
- true about the representation / unproven about the represented reality.

Contradiction is often a diagnostic of an omitted variable, an inappropriate boundary, an unrepresented timescale, or a category that has been made too coarse. It is not automatically an error to be erased.

---

## 8. How to investigate without turning escape into a ritual

This doctrine must never become a compulsory checklist, a daily ceremony, a fixed template, a vocabulary test, or an endlessly repeated demand to “question assumptions.” Any of those can become a new box: a process that performs openness while protecting its own categories from challenge.

There is no mandatory sequence of questions and no prescribed cadence. Instead, use the doctrine as a living permission structure:

- when a conclusion feels inevitable, ask what made it feel inevitable;
- when an obstacle looks fundamental, identify the layer in which it is fundamental;
- when a measure becomes persuasive, ask what it cannot see;
- when a tool dictates the available actions, separate the task from the tool’s affordances;
- when an explanation becomes too coherent, look for the cost of its coherence;
- when a proposal sounds radical, ask whether it merely recombines inherited categories;
- when a result is surprising, ask whether the surprise is data, a flaw, or evidence that the ontology is incomplete;
- when a new frame becomes successful, turn the doctrine on that frame as well.

These are not commands to be performed mechanically. They are directions of attention to invoke proportionately, especially at points of apparent certainty, repeated failure, sudden success, or irreversible commitment.

The doctrine should reduce needless ritual, not add it. If a practice produces no new distinction, no alternative causal account, no changed prediction, and no changed action space, it may be theater rather than conceptual escape.

---

## 9. Constructive reinvention: an alternative must sometimes be built, not merely imagined

There is a difference between saying that a category could be replaced and discovering what follows when it actually is replaced. A system that only lists alternatives will usually return to its inherited implementation, vocabulary, and tools unchanged. Conceptual escape becomes real only when an alternative representation is made concrete enough to create consequences that the incumbent representation could not cheaply express.

Accordingly, at every **meaningful milestone**—a point at which a representation, architecture, metric, dataset, safety boundary, or claimed limit has influenced subsequent choices—the work must perform one bounded **missing-world intervention**. The intervention is not a request for arbitrary novelty. It is a requirement to construct, specify, or rigorously rule out one materially different possibility chosen independently of the dominant solution.

The intervention may, for example:

- remove a primitive that the current system treats as indispensable;
- replace an object by a program, relation, field, proof, distribution, or transformation;
- merge two components that were assumed to be separate, or separate two concerns that were assumed inseparable;
- reverse a causal direction, change the unit of work, or make a hidden timescale explicit;
- rebuild a constraining language, library, protocol, or measurement from first principles as a disposable bootstrap rather than a governing ontology;
- introduce an observation that the dominant metric cannot represent;
- formulate a counterfactual world in which a current limitation never existed, then identify the smallest responsible experiment that distinguishes that world from the incumbent one.

The output must be more than a slogan. It must be one of the following:

1. a minimal executable or formal prototype;
2. a precise alternate representation with independently checkable consequences;
3. a falsifiable experiment or proof obligation;
4. a documented reason why construction is currently unsafe, ill-posed, or blocked by a specific missing capability.

This is deliberately bounded. It does **not** mean that every component must be discarded on every iteration, nor that the current work must be paused until every imaginable ontology is explored. It means that the dominant path is never allowed to become the only path merely because it already exists.

The crucial first question is therefore not merely:

> How can this component be improved?

It is also:

> Why does this component, category, language, library, metric, boundary, or sequence need to exist at all?

If the answer is only historical convenience, tool availability, or habit, its replacement deserves construction—not blind adoption, but a bounded, honest attempt.

### 9.1 The novelty register: preserve dormant possibilities without promoting them to facts

An alternative that does not immediately win on the current objective may still reveal a future route, a missing variable, a reusable proof technique, or a later change of scale. Do not discard it merely because the incumbent benchmark did not reward it today.

Maintain a durable **novelty register** separate from the dominant roadmap. Each entry must record:

- the inherited framing that was challenged;
- the alternate ontology or representation;
- the smallest constructed artifact, formal statement, or experiment;
- exact local evidence and its provenance;
- what has and has not been independently verified;
- the local scope of any conclusion;
- the specific future contexts in which the idea might become useful;
- dependencies, safety implications, and possible harms;
- the condition that would revive, refute, merge, or retire the entry.

The register is not a graveyard for vague ideas and not a catalogue of claims. Its entries are **dormant hypotheses with provenance**. A recorded possibility is neither true nor useless. It remains available for re-evaluation when another representation, tool, corpus, scale, or causal relation changes.

### 9.2 The anti-hallucination boundary

Radical construction is not permission to state desired outcomes as achieved outcomes. Every entry and every intervention must distinguish four layers:

| Layer | Permitted statement |
| --- | --- |
| Exact fact | A claim directly established by a reproducible proof, measurement, or independently checkable execution. |
| Derived inference | A conclusion that follows from stated facts under explicit assumptions. |
| Hypothesis | A proposed causal, mathematical, or architectural explanation that still needs a discriminating test. |
| Speculative possibility | A deliberately preserved idea whose usefulness, feasibility, or even correct formulation is unknown. |

Never upgrade a statement from one layer to another because it is exciting, elegant, radical, or aligned with a desired result. Conversely, never reject a speculative possibility merely because the present system cannot yet measure it. The discipline is to keep the boundary visible while preserving the route.

### 9.3 Preventing the new rule from becoming a new box

The missing-world intervention is mandatory as an outcome at meaningful milestones, but its form, source, and timing must remain revisable. It may be generated by a person, a model, a formal system, a hostile review, an unrelated discipline, a historical failure, an absence in the data, or a physical constraint. Reusing the same ritual, source of inspiration, or preferred kind of novelty repeatedly is itself evidence that the doctrine has narrowed into a new box.

An intervention that produces no changed distinction, no new testable consequence, and no new action space may be recorded as a failed attempt—but it must not be celebrated as innovation.

---

## 10. Refutability without imprisonment by the current frame

Escaping a frame does not license ungrounded invention. A new representation earns attention when it changes something consequential and exposes itself to a meaningful possibility of failure.

For a proposed reframing, seek to make clear:

1. **What distinction has changed?**
2. **What previously impossible or invisible action becomes thinkable?**
3. **What different consequence should follow if the reframing is useful?**
4. **What observation, intervention, or coherent counterexample would weaken it?**
5. **What must remain invariant for the proposal to remain responsible?**

These prompts are not a required form. They are a minimum intellectual discipline against novelty theater.

The ideal test does not merely ask whether a proposed improvement wins on an inherited metric. It asks whether the new ontology reveals a different observable consequence, including consequences that the old frame could not have formulated in advance.

---

## 11. Learn from failures, absences, and discarded material

The box often decides that only successes are informative. This creates a severe loss of structure.

Treat failures, near-misses, contradictions, delays, discarded traces, non-events, and irreproducible outcomes as possible evidence about the representation. Ask:

- What category made this result appear irrelevant?
- What relation would make the discarded material informative?
- Does the failure reveal a boundary condition rather than a lack of ability?
- Is the absence genuine, or is it an absence created by the observation process?
- Could a record of failure be a reusable transformation, constraint, certificate, or map of forbidden regions?

This is not a demand to preserve everything forever. It is a demand not to discard information solely because the inherited objective did not reward it.

---

## 12. Scale, time, and reversibility

Many boxes are produced by a single timescale or a single notion of reversibility. A choice that looks optimal immediately may create a distant lock-in. A choice that looks costly locally may change the reachable future. A historical record may be evidence of the world, evidence of the observer, or evidence of the path by which the world became observable.

Therefore consider, without assuming a preferred answer:

- short versus long horizons;
- local versus distributed effects;
- reversible versus irreversible commitments;
- static versus adaptive environments;
- direct versus mediated observation;
- first-order versus feedback effects;
- apparent scarcity versus access constrained by representation;
- historical data versus counterfactual structure.

The task is not to model every scale. It is to avoid treating the currently visible scale as the only real one.

---

## 13. The doctrine as a self-revising object

This document is not outside the box by virtue of its title. It is itself a representation, a vocabulary, a set of distinctions, and therefore a potential box.

It must be challenged whenever it:

- makes “outside” into a badge rather than a change in possibility;
- rewards novelty merely for being unfamiliar;
- turns doubt into paralysis;
- treats disruption as intrinsically good;
- privileges its own vocabulary over a better future vocabulary;
- uses “unknown unknowns” as an excuse not to learn from available evidence;
- becomes a ritual that substitutes for action;
- suppresses a useful conventional solution because it is conventional;
- frames safety, care, truthfulness, or accountability as obstacles to escape;
- mistakes maximal conceptual distance for maximal real-world value.

If any of these occur, revise, suspend, or replace the doctrine. A doctrine that cannot be revised has become exactly the kind of box it was meant to expose.

---

## 14. A compact orientation

When no longer sure how to proceed, the following orientation may help:

> Do not ask only how to improve the thing named by the current frame. Ask why that thing, that name, that boundary, that causal direction, that unit, that constraint, and that measure exist—and what changes if any one of them is no longer primary.

Then preserve the ability to be corrected by consequence.

---

## What this doctrine does **not** authorize

This doctrine does **not** authorize:

- unsupported claims of discovery, superiority, inevitability, or safety;
- ignoring evidence because it is inconvenient to a preferred story;
- rejecting verification, reproducibility, provenance, or honest uncertainty;
- violating justified safety, ethical, legal, or consent-based constraints;
- treating affected people or systems as disposable experimental material;
- concealing costs, harms, failures, or uncertainty behind the language of innovation;
- confusing novelty with value, complexity with depth, or disruption with progress;
- refusing ordinary solutions when they are the best-supported responsible choice.

The doctrine authorizes conceptual freedom only together with intellectual honesty, responsibility for consequences, and continuous openness to correction.
