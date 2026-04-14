# CohenDenham2019

_Generated from: https://www.sciencedirect.com/science/article/pii/S2452310018301082_

## Page 1

# Whole animal modeling: piecing together nematode locomotion

**Abstract**
With a reconstructed and extensively characterized neural circuit, *Caenorhabditis elegans* is a fascinating model system for the study of neural circuits and behavior. Here, we review the recent progress in the study of locomotion in this animal from a systems perspective. We discuss how complementary approaches, from network science, through dynamical systems to biomechanics are transforming the current understanding of this system into a unified whole animal description. This transformation has been achieved through the integration of mechanistic studies and decompositional approaches: on the one hand, mapping the components of the system and their functions and on the other hand, providing qualitative and quantitative methods to probe the physical basis of locomotion, motor behavior, neural dynamics, and structure–function relation in neural circuits.

Current Opinion in Systems Biology 2019, **13**:150–160
This review comes from a themed issue on **Systems biology of model organisms**
Edited by Denis Dupuy and Baris Tursun
For a complete overview see the Issue and the Editorial
Available online 12 December 2018

2452-3100/© 2019 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY license (

).

## Introduction

Deciphering the neural control of behavior is a systems challenge that requires integration of structure, function, and dynamics across scales, from the gene to the behaving animal. Here, we review recent progress in our understanding of a subset of behaviors in the model organism *Caenorhabditis elegans*, with a focus on progress and open challenges for systems modeling.

In comparison to humans, the anatomical structure of nematodes is remarkably simple. *C. elegans* has a fully mapped and invariant cell lineage; the adult hermaphrodite has 959 somatic cells including 95 body wall muscles and precisely 302 neurons (with a fully mapped connectome \[1–4]). The slender, 1 mm long animal is lined with muscles that are controlled by a relatively distributed nervous system.

The compact and small nervous system of nematodes precludes the complex organization of the human brain, as well as many neural functions. Nematodes have no visual or auditory system or any obvious evidence of neuronal representations of complex spatial or other features, nor do they possess limbs or any means of complex communication. Yet, they are fully functioning, free-living animals that can forage for food, escape predation, and effectively navigate complex physical terrains and rich chemical environments. These are the elementary functions of nervous systems that are common to most animals. Not surprisingly, these tend to be heavily reliant on locomotion. *C. elegans* and its neural control of locomotion offer us a window to focus on the principles and mechanisms behind these essential behaviors.

In the lab, the movement of *C. elegans* is studied predominantly on the surface of agar gels. Here, nematodes lie on one side of their body (either left or right) and undulate in the dorsoventral plane, propagating waves from head to tail and pushing against the substrate to generate forward movement. Occasionally, animals will reverse the direction of their undulations to move backward. Backward movement is often implicated either in escape or in reorientation. To turn, animals can either gently steer by biasing head and neck undulations, with the body following suit \[5–8], or turn more sharply, by deeply bending the body into stereotypical omega or delta body shapes \[9–11] and emerging in a new orientation. Deep bends that mediate an escape response can generate a near 180° reorientation \[12], whereas other turns, typically observed during area-restricted search, foraging, and chemotaxis, generate more broadly distributed reorientations \[10,11,13]. Studies in other physical environments demonstrate the ability of the animal to robustly adapt its waveform and kinematics to the surrounding environment \[14–17].

## Common topological structures across nervous systems

The known structure of the *C. elegans* nervous system provides an excellent starting point for linking neural

## Page 2

dynamics with behavior. While the size and organization of vertebrate and nematode nervous systems are vastly different, some broad principles appear to unify them, including a hierarchical structure with high clustering and a small-world organization \[18]. Hierarchical structure refers to a multiscale architecture with nesting of modular subnetworks, whereas high clustering corresponds to local features, often manifesting in a large proportion of connected three-node subcircuits \[19–22]. Studies also find similar topological features connecting low-level (microcircuit) clusters or network motifs with the modular high-level (brain-wide) architectures. Specifically, both small-world and rich-club organization have been identified in the wiring of the *C. elegans* nervous system \[18,23] and high-level connectivity in the human brain \[24].

In small-world networks, most neurons can be reached from every other neuron through a small number of synaptic connections; the short distances or path lengths between neurons are often mediated through high-degree hub nodes. Rich-club networks contain hub nodes that are themselves disproportionately interconnected, a topology that may serve to enhance the robustness and resilience of certain network functions. In both *C. elegans* and the human brain, these features of network topology suggest a great deal of coordination between specialized subcircuits. As in the human brain \[24], so in *C. elegans*, the rich club has been linked with whole brain communication \[25]; in *C. elegans*, however, the prevalence of locomotion interneurons in the rich club (Figure 1a) indicates a direct and strong coupling between sensory information, global brain dynamics, and motor behavior \[25,26]. This is not surprising given the importance of locomotion for the survival of the animal.

The mapped connectome and ability to target, manipulate and record from identified neurons in *C. elegans* have meant that most neuron classes are identified with specific subcircuits and motor behaviors, including a number of locomotion interneurons driving forward and backward locomotion \[25,27–30]. These locomotion interneurons act as on/off switches to gate distinct forward and backward locomotion circuits in the ventral nerve cord (VNC). Importantly, all but one of the set of rich-club neurons identified corresponds to locomotion interneurons, pointing to the essential role of this circuit for the survival of the animal.

### From structure to function

Network theory offers tools for inferring function and dynamics from topology. However, to learn about *C. elegans* dynamics from this advancing mathematical field, we must tread carefully. For example, it is tempting to interpret the feed-forward network motif, so prevalent in the *C. elegans* sensory system, as a logical AND gate \[19], but there is as yet no experimental demonstration of corresponding dynamics. Similarly, the short path lengths, coupled with the small number of hub nodes, are often interpreted as mediating rapid communication and synchronization among nodes or brain areas, but a recent simulation study shows that the governing time scales of dynamics on the *C. elegans* connectome are not a straight forward consequence of either path length or in-degree (i.e. the number of presynaptic connections) \[31]. On the one hand, that study suggests that the *C. elegans* circuit is in some sense optimized for fast, coordinated control but on the other hand that we still lack the network theoretic tools to pin down the corresponding structure—function relation. An alternative approach is to consider the connectome as a test bed for evaluating network theoretic tools and the validity of their assumptions.

Controllability of the *C. elegans* nervous system is one example of this alternative approach. Yan et al. \[26,32] use control theory to ask which muscles can be independently controlled and which neural nodes and pathways control specific muscles or specific motor behaviors, focusing on the locomotory response of the animal to gentle touch. Within the framework of the model assumptions and subject to the limitations of the available connectome \[2,3], the authors find that 89 of the 95 body wall muscles are independently controllable. Furthermore, elimination of specific classes of neurons only rarely leads to a reduction in the number of independently controllable muscles, indicating that this level of controllability is robust: In this model, only 12 classes of neurons appeared to reduce controllability, and all but one of these loci of controllability are identified with the locomotor circuit \[23,32].

Naively, the ability to independently control 89 body wall muscles suggests the potential for rich neuronal dynamics and a vast space of possible patterns of muscle activation. In fact, although the worm exhibits a rich repertoire of motor behaviors, these behaviors appear highly coordinated and involve smooth propagation of muscle activation along its slender body. Given the strong assumptions of the model — linear dynamics subject to a single governing time scale — the level of controllability is best interpreted as an upper bound. The apparent discrepancy with observed low-dimensional behavior points to the importance of additional factors in determining controllability and calling for further advances in network control theory to cope with these more general classes of dynamics \[33].

One regime in which a linearity assumption may be instructive is in the immediate neighborhood of a bifurcation. Large-scale anatomically grounded computational models of the human brain best capture empirical data of spontaneous resting activity when the system operates at the critical point of an instability \[34]. In such models, intrinsic fluctuations

## Page 3

|**Neuron Class**|**Function/Description**|**Connections**|
|-|-|-|
|AIB|First-layer interneurons, integrating sensory inputs and locomotion gates|Synapses onto VNC motoneurons and other locomotion interneurons|
|RIB RIB|Second-layer interneurons combining thermo- and chemo-sensory integration with motor control|Extensive outputs to head motoneurons|
|RIA|Second-layer interneurons combining thermo- and chemo-sensory integration with motor control|Extensive outputs to head motoneurons|
|DVA|Proprioceptive head interneurons; fine tune forward accelerations and reversals|Regulate both head and tail touch circuits|
|AVB|Ventral nerve interneurons that gate forward locomotion|Innervated by anterior sensory and first- and second-layer interneurons|
|AVA|Ventral nerve interneurons that gate backward locomotion backward locomotion|Innervated by anterior sensory and first- and second-layer interneurons|
|AVD|Ventral nerve interneurons that gate backward locomotion|Innervated by anterior sensory and first- and second-layer interneurons|
|AVE AVE|Ventral nerve interneurons that gate backward locomotion backward locomotion|Innervated by anterior sensory and first- and second-layer interneurons|
|PVC|Integrate mechano- and chemo-sensory inputs in the tail in the tail|Help gate forward locomotion|

Insights from the *C. elegans* connectome into locomotion control. (a) The *C. elegans* rich club features nine classes of neurons (circles) that are prominent in sensorimotor decisions and motor commands (adapted from Towlson et al. \[23]). AIB neurons double as first-layer interneurons, integrating over sensory inputs and as locomotion gates, synapsing onto VNC motoneurons as well as other locomotion interneurons. RIA and RIB are second-layer interneurons combining thermo- and chemo-sensory integration functions with motor control, e.g. through extensive outputs to head motoneurons. DVA head interneurons are proprioceptive and fine tune forward accelerations and reversals during locomotion; they also regulate both head and tail touch circuits. Ventral nerve interneurons that gate forward (AVB) and backward (AVA, AVD, and AVE) locomotion are innervated predominantly by anterior sensory as well as first- and second-layer interneurons. PVC interneurons integrate mechano- and chemo-sensory inputs in the tail and help gate forward locomotion. Line widths represent the number of chemical (black) and electrical (red) synapses. All self-connections denote intraclass synapses (between left and right neurons). (b) Simplified locomotion motor circuit of the *C. elegans* VNC, depicted as repeating neuromuscular units, has served as a basis for a number of computational models. The model circuit depicts mirror images of forward and backward locomotor circuits. Motoneurons of classes VA, VB, VD (DA, DB, DD) innervate ventral (dorsal) muscles. (c) One of six repeating neural units derived from the connectome (adapted from Haspel and O’Donovan \[67], updated by Gal Haspel, personal communication), including AS class motoneurons in addition to motoneuron classes in Figure 1b. Neuronal placement is determined by their muscle connectivity, and the repeating structure was obtained by selecting sufficiently strong and sufficiently repeated connections along the VNC. Gap junctions are purple. Line widths for chemical synapses represent contact strength. VNC, ventral nerve cord.

## Page 4

spontaneously trigger waves of activity — dynamical excursions to one of multiple accessible attractor states — whose form is largely dictated by the anatomical connectivity. It is therefore in this rest state that the anatomical and functional connectivities of the system can be best explored simultaneously \[34]. Whether the same reasoning applies to *C. elegans* remains an open question and could be explored further by explicitly linearizing the dynamics near a critical point.

Support for the crossing of a bifurcation as the worm transitions from quiescence to arousal comes from brain-wide imaging studies of *C. elegans* \[35]. Brain-wide imaging has now been performed both at rest and in freely moving animals \[36–38], revealing complex patterns of both spontaneous and evoked activity of head neurons \[36]. Moreover, Nichols et al. \[35] identify global attractor states corresponding to different motor programs (and a fixed state for quiescent behavior), generalizing the concept of gating different motor behaviors through a single locomotion interneuron to brain-wide distributed networks, again supporting a picture of competition between a set of attractor states.

One might expect the highly recurrent topology of the circuit to give rise to sustained oscillations under external stimulation. However, simulations of a virtual connectome subject to stimulation of one node at a time \[39] yielded stationary network states except under very strong current input to a small number of neurons. Importantly, this and follow-up simulation models also assume linear neurons with a single governing time scale \[39,40]. Any nonlinearities in these models arise due to the synaptic and gap junctional connectivity. Therefore, if, as these models suggest, the structure of the connectome does not lend itself to spontaneous oscillations, this offers a further indication of the importance of nonlinearities in generating oscillations in *C. elegans*.

### Top-down insight from behavior

The aforementioned connectome-wide simulations \[39,40] suggest that whatever the fine control of the neural circuit, the number of stable global modes supported by this circuit is low. A complementary approach is to consider the postural modes of the animal. Assessing free behavior is particularly interesting as it is not limited to stable states. Thus, it may initially seem surprising that nearly all postures of this nematode on agar can be well approximated by a very low (four to five) dimensional space \[11,41]. Furthermore, the dynamics of these postures can be mapped to a small number of attractor states, corresponding to distinct classes of motor behaviors, such as forward movement, backward movement, and turning. Together, these results point to strong organizing principles of animal behavior: Whereas the anatomical connectome may point to up to 89 muscles that may be independently controlled, suggesting unfathomable complexity, the observed repertoire of behavior is in fact very limited, occupying specific manifolds within a low-dimensional ‘state space’.

The ability to describe up to 95% of worm postures using a small number of principle components, dubbed eigenworms \[11,41], has proved to be a powerful tool in the *C. elegans* community. Observing behavior through the lens of eigenmodes has led to the identification of a new turning behavior dubbed delta turns that occur independently of omega turns (stereotypical of the escape response), suggesting that these distinct motor programs are produced by distinct pathways. Eigenworms and other low-dimensional representations of posture have been used for genetic fingerprinting of different wild-type and mutant *C. elegans* strains \[11,42,43]. Another application has been a powerful visualization tool for neural recordings \[25,38], revealing a tight correspondence between global neural and behavioral states \[25,38]. Finally, Li et al. \[44] recently applied a machine learning approach to synthetic posture and trajectory generation; a neural network, using a brief seed of postures as input (in dimensionally reduced form), generated worm postures that were subsequently used to generate realistic trajectories in space. Such results lend further credence to the conjecture that low-dimensional neural dynamics can account for observed motor behaviors. Furthermore, the ability to synthetically phenocopy trajectories of different mutant strains provides a useful tool not only for genetic fingerprinting but potentially also for mechanistic models of neural control.

### The locomotor circuit

The small size and relative simplicity of the *C. elegans* connectome, together with powerful genetic, molecular, and optical tools, allow for a detailed analysis of neuronal connections and functions \[1–3,45,46]. The VNC runs along the body and contains eight classes of motoneurons, each with characteristic anatomy, that drive ventral and dorsal body wall muscles \[27,47]. Early inspection of the connectivity highlighted key departures from familiar network motifs in other locomotor circuits \[48–51]: The circuit in the VNC is dominated by excitatory neurons and gap junctions \[52] rather than inhibition, raising questions about the mechanisms of pattern generation. In contrast, models of the head circuit \[53,54] and recent experimental reports \[55,56] indicate that the head locomotion circuit is dominated by inhibition.

Systematic neuronal ablations and more recent *in vivo* calcium imaging and optogenetic experiments associated distinct classes of motoneurons with distinct motor behaviors \[12,27,57–60]. Of the eight classes of motoneurons in the ventral nerve cord, forward movement (consisting of dorsoventral activation that flows from head to tail) requires VB and DB cholinergic excitatory

## Page 5

motoneurons (B-type for short, Figure 1). The backward locomotion circuit approximately mirrors the forward circuit, supporting the flow of activation from tail to head, via VA and DA (A-type) excitatory motoneurons. The only \gamma-amino butyric acid (GABA)ergic motoneuron classes in the VNC, VD, and DD modulate undulatory locomotion but do not appear essential for crawling \[51,57,60,61]. AS cholinergic motoneurons contribute to locomotion, but recent experiments suggest that they are not essential for rhythm generation in either forward or backward movement \[60], and VC cholinergic motoneurons have been implicated primarily in egg laying \[59].

With this class assignment in hand, early descriptions of the VNC sought to view it as a set of repeating subcircuits. As the number of neurons differs from class to class, a parsimonious simplified structure was proposed, with a single neuron of each class per repeating unit \[1,27,47] (Figure 1b). Although computational models using such simplified wiring diagrams have yielded significant insights into pattern generation and neuro-mechanics, \[48,50,51,62–66], the reduced complexity of repeating structures risks the loss of key degrees of freedom that underpin behaviorally important forms of neural dynamics, especially if those involve previously overlooked classes of motoneurons that have a role in distributed computation. In particular, the apparent inability of such minimal representations to endogenously generate distributed locomotory patterns motivated a systematic connectomic analysis that accounted for the varying cardinality of each class \[67,68] (Figure 1c). The resulting set of repeating units also include a previously uncharacterized motoneuron class (AS motoneurons), suggesting a role in the control of locomotion.

## Feed-forward and feedback models of locomotion

The elegant sinuous gait presents three fundamental questions \[65,69]: How are rhythmic oscillations generated? How are they coordinated across opposite (dorsal and ventral) muscles? And, how is their propagation coordinated along the animal? Recent experiments provide the first hints that B-type neurons in the forward locomotor circuit may support distributed oscillations along the body \[61,64], with possible coupling to a pacemaker in the head \[61,64]; meanwhile, a number of theoretical studies have asked whether peripheral control may provide a pathway for pattern generation \[14,40,51,63]. Whether centrally or peripherally generated, there is strong experimental evidence that the entrainment or phase coordination of these oscillations requires proprioceptive feedback \[61,64,70].

As noted previously, connectome-based disembodied models with linear neurons \[39] have struggled to generate fictive oscillations. In contrast, Olivares et al. \[71] considered bistable neural dynamics in a simplified connectome \[67,68] (Figure 1c). This model exhibits distributed oscillations that are generated by three classes of motoneurons \[71], including the newly conjectured AS motoneurons \[67]. Importantly, however, the oscillatory motifs (obtained in this model through an evolutionary search algorithm) rely on extensive inhibition. This work reinforces the difficulty of generating endogenous oscillations without either pacemakers or sensory feedback. As a follow-up to this modeling study, direct recordings of AS activity appear to rule out their role in pattern generation, instead implicating them in the locomotion interneuron gating circuit and in modulating the kinematics of undulations \[60].

Pattern generation is much more easily achievable in computational models of proprioceptive control \[49–51,63,72–74]: Local bending of one side of the body triggers stretch activation of the opposite B-type motoneurons. Either posteriorly facing \[75] or anteriorly facing \[64,70] proprioceptive fields can mediate robust undulations when coupled to a pacemaker in the head. But, body undulations can be generated in models even in the absence of head oscillations \[51,63]. Bryden and Cohen \[50] demonstrated that adding local to distal proprioception enhances the robustness of undulations, and Boyle et al. \[51] and Denham et al. \[63] showed that local proprioception is sufficient for crawling on agar-like substrates, but not for swimming in low viscosity liquids. Together, these models predict that the spatially extended field manifests behaviorally in more dilute media. As we see below, in these models, the frequency and wavelength of undulations are tightly coupled through the proprioceptive integration of the body posture which, in turn, depends on the time taken by the biomechanical body to bend in its physical environment.

Interestingly, proprioceptively driven models have long required strong nonlinearities in B-type excitatory motoneurons \[49–51,64,65,76]. Bryden and Cohen \[50] required strongly nonlinear stretch receptor conductances in their model of B-type motoneurons, effectively yielding on-off (resting and upstate) membrane potentials. Following this work, direct electrophysiological recordings gave the first direct evidence of bistable motoneurons in *C. elegans* (the RMD neurons in the head) \[28], inspiring Boyle et al. \[51] to consider bistability in B-type motoneurons in their model. More recently, electrophysiological recordings provided direct evidence for bistability in both A- and B-type motoneurons \[77]. The argument for bistability in B-type motoneurons is strengthened by intuition from engineering principles: First, the hysteresis inherent in the proposed switching mechanism provides robustness to fluctuations; second, the distinct on/off states allow for efficient alternating action of opposing muscles \[51,65].

## Page 6

A further prediction that arose from the model of Boyle et al. \[51] is a resetting mechanism that coordinates robust antiphase activation of dorsal and ventral muscles. Whereas the conventional intuition is that D-type inhibitory motoneurons suppress muscles of the noncontracting side, this model suggests an additional rhythm generating role for D-type neurons through the inhibition of excitatory motoneurons on the ventral side (Figure 1b). This requirement arises directly from the bistability condition in B-type neurons: Without ventral inhibition, the bistable switch could lead to pairs of ventral and dorsal motoneurons being on (or off) at the same time, thus disrupting or even freezing the undulatory wave. Inhibition on one side of the body suffices to avoid such a failure, by imposing dorsoventral coordination. Importantly, the ability to discern this neuronal reset depends on the biomechanics of the locomotion: The model predicts that the contribution of neural inhibition would be masked on agar but has growing importance for rapid, swimming undulations in less viscous fluids.

How do forward and backward locomotion patterns differ? Backward locomotion is a rarer transient behavior, lasting at most a few undulations \[10]. A combination of studies including optogenetic, electrophysiological, and calcium imaging techniques ascribe this effect to an imbalance in the locomotion interneuron circuit \[55,60,78,79]. Haspel and O’Donovan \[67,68] give the first hint of subtle but systematic asymmetries between the forward and backward microcircuits of the VNC. Recent experiments point to slow pacemaking A-type motoneurons in the backward circuit \[80]. The picture that emerges assigns three roles to A-type neurons: in gating \[64,78,79], in endogenous pattern generation \[80], and in mediating proprioceptive feedback \[80].

How does the head control and modulate undulations along the body? While neuromechanical models of the body appear fully capable of realistic undulations \[51,63], the circuit that orchestrates, modulates, and switches between different motor programs resides in the head \[25,35,36,46]. During forward locomotion, the body clearly follows the head, and biased head oscillations that track sensory inputs can thus steer the locomotion \[7,8,72]. Omega turns are similarly initiated in the head \[11,12] but require the body to actively follow, for example, in one model, through a traveling wave of suppressed proprioception \[74]. In a separate neuro-mechanical model, a worm lacking a VNC circuit can still follow the head on agar, although severely uncoordinated \[73]; this model relies on proprioceptively driven oscillations generated by SMD and RMD head motoneurons. Understanding the interface between head and body circuits is complicated by the possibility of mismatched frequencies and phases between head and body oscillations \[36,61]. Proprioceptive mechanisms are a strong candidate for coordinating the head and the body. Dorsal SMD (SMDD) neurons (implicated in steering) have now been experimentally demonstrated to be proprioceptive \[8]. Spatially extended neural processes (posteriorly facing in SMD and anteriorly facing in the anterior-most VB motoneurons) may inform the proprioceptive range and mechanisms of such coordination \[1,4].

## Biomechanical and neuronal substrates of gait adaptation

When *C. elegans* is placed in a low viscosity liquid, its elegant, slow and sinuous crawling gait is replaced by rapid, long wavelength, and high amplitude undulations, dubbed swimming \[14,81]. Berri et al. \[14] showed that swimming and crawling constitute a single biomechanical gait that is smoothly modulated as a function of the resistivity of the environment \[15,62,65,82,83]. Boyle et al. \[51] demonstrated that this form of gait modulation is a natural outcome of proprioceptively controlled locomotion: A single fixed-parameter and ‘headless’ model worm can produce both swimming and crawling, as well as undulations in intermediate Newtonian, linear viscoelastic and obstacle-rich environments. This form of gait modulation can be summarized by a smooth relationship between kinematic parameters: The faster the undulations, the longer the wavelength and amplitude of undulations along the body.

Key to understanding the interplay between the neural dynamics and biomechanics underpinning this modulation are the material properties of the body. In the case of the worm, dissecting the relative roles of essential contributing factors has relied extensively on biomechanical models, often iterating closely with experiment. Contributing factors include internal pressure and bulk elasticity \[84], elasticity of the cuticle and muscles \[15,85–89], internal viscosity of the body \[85,87,88], and the activity-dependent regulation of muscle tone \[90]. Some models (with various levels of abstraction) have also used the worm as a platform to characterize liquid flow and viscoelastic properties of Newtonian and complex fluids \[14,82,91–93]. The aforementioned approaches to characterizing material and fluid properties require only a model of the body and surrounding environment (without neural control) to solve the equations of motion. For example, by periodically forcing the mechanical model with different waveforms and simulating the dynamics in different fluid environments, it can be shown that the modulation of waveform as a function of fluid viscoelasticity provides important kinematic advantages (minimizing power and enhancing locomotion speed) \[89].

An important test of biomechanical models lies in their capacity to advance our understanding through the integration of neural control and biomechanics in a single, whole animal model. At the software level, this

## Page 7

calls for flexible interfaces between neural, muscular, and mechanical components of the model to support plug-and-play experimentation with different forms of neural control \[63]. Furthermore, as the number of control parameters grows in a model, computational efficiency becomes of paramount importance. Several of the models mentioned previously have the capacity to support a variety of simulation experiments, including extensive parameter sweeps, leading to fundamental insights into this system. For example, Denham et al. \[63] revisited the constraints on material properties in a model of proprioceptively driven control (akin to Ref. \[51]) integrated into a viscoelastic shell model \[89]. The model demonstrates how body elasticity and external drag reduce to a single universal parameter that describes the kinematics of the motion in Newtonian media and to two parameters in the case of linear viscoelastic media. Furthermore, the model predicts that only a limited range of effective body elasticity can support the full range of observed gait modulation. Importantly, the capacity of this neuromechanical model to address gait modulation benefits from the exact mathematical formulation of the model \[89], departing from previous formulations that linearize the equations around a point in parameter space.

Integrated neuromechanical models also provide a framework to test hypotheses about neural control and to identify candidate targets of internal modulation. For example, fish can independently alter activation frequency and duty cycle of centrally generated rhythms, but only some neural pathways of such modulations have been identified. *C. elegans* too can modulate its locomotion speed and some kinematic parameters. Denham et al. \[63] asked what gait modulation would look like under proprioceptive control, focusing on the three natural targets of modulation: a change in elasticity due to the modulation of muscle tone, the activation threshold of B-type motoneurons, and the spatial range of the proprioceptive field. Targeting locomotion interneurons AVB (or AVA) or the AVB-B (or AVA-A) gap junctions in the forward (or backward) circuit directly maps to a modulation of threshold in this model. Following the aforementioned reasoning, the internal modulation of mechanical properties such as body elasticity mirrors that of environmentally (or externally) imposed gait modulation, yielding a positive frequency—wavelength correlation. In contrast, internal modulation of neural parameters gives rise to the opposite relation: The higher the frequency, the lower the wavelength of undulations. This signature of internal gait modulation in the form of an inverse wavelength—frequency relation is specific to proprioceptive pathways of control and is therefore unlikely to be obtained by a modulation of a central pattern generator. The result therefore lends itself to a number of direct experimental predictions that may shed light on the respective roles of central and peripheral control in *C. elegans* locomotion in the forward and backward circuits and may help identify neural pathways and targets for their modulation. For example, if A-type (backward locomotion) but not B-type (forward locomotion) neurons act as pacemakers, the modulation of their respective circuits should yield distinct kinematic signatures.

## Discussion and future outlook

Animal locomotion is a fascinating playground for exploring the interplay among the genetic, molecular, and biophysical contributions in neurobiology; the structure, function, and dynamics of neural circuits; and the biomechanics of motor behavior. The relative simplicity of *C. elegans* has allowed for an unprecedented level of characterization and an ever-growing experimental toolkit; together, these have facilitated interdisciplinary discourse, leading to significant advances in our understanding of the locomotion system and a window into understanding the ‘state of mind’ of the worm and organization of its nervous system more generally. This review has highlighted the contributions of theory and data-driven models. A recurring thread in this review has been to highlight how different formulations of the problem can serve, not always to generate testable predictions but rather to allow for the testing and, in some cases, falsification of model assumptions.

The neurodynamics of the *C. elegans* head circuit maps onto locomotor states and is well described by competition and transitions among a small number of attractors. Transitions among motor programs are clearly evident in the switching of locomotion interneuron states that drive the motor circuits along the body. The most prevalent motor behavior — forward locomotion — is well described by a single biomechanical gait that adapts smoothly and continuously to the external physical environment. Internal regulation of muscle tone and modulation of the neural circuit allow for an impressive range and specificity of kinematic control across a range of motor programs. There is now compelling evidence for proprioceptive control of the A- and B-type excitatory motoneurons of the VNC, although specific stretch receptor proteins are yet to be identified and characterized. Recent experiments are also beginning to unravel the possible roles of distributed pattern generation along the body. The interplay between central and peripheral control is therefore an exciting topic of ongoing and future investigation. Unlike the VNC, the head circuit appears to be dominated by inhibition, and the connectome suggests a number of candidate circuits for central pattern generation. As data accumulate, models are beginning to address the sensorimotor control of oscillations in the head, the role of proprioception, and the coordination between the head and the body.

## Page 8

This review has focused on the connectome, neural dynamics, and behavioral aspects of locomotion, excluding the large body of research on the genetic specification of behavior \[94], neurophysiology, and biophysical properties of neurons and muscles \[52,78,79] and exciting advances in our understanding of extrasynaptic communication networks \[95], with their contributions to remodeling of neurons and neural circuits.

Rapidly growing computational power, tools, and resources are facilitating a step change in the generation and analysis of big data, including static networks, behavioral, and brain-wide imaging data, or high-throughput simulation. Simulation frameworks such as Openworm \[96–98] and *Si elegans* \[99,100] are pushing the computational limits. Aimed principally at emulating the biological system, these frameworks are designed to provide unprecedented anatomical and molecular level detail of the biophysics and mechanics and are already leading to simulations of sensorimotor behavior in embodied, situated, and freely behaving model worms \[100]. The plurality of modeling frameworks and data will allow a variety of modeling questions to be addressed, benefiting from validation across different platforms. Future progress is therefore increasingly relying on plug-and-play software environments, in which anatomically or biophysically detailed model components — and data — can be seamlessly switched on or off or interchanged with simpler, theory- or hypothesis-driven models.

## Conflict of interest statement

Nothing declared.

## Acknowledgements

N.C. acknowledges funding from the EPSRC (EP/J004057/1 and EP/S01540X/1). The authors thank Thomas Ranner, Felix Salfelder, Gal Haspel, Ian Hope and Samuel Braunstein for useful discussions.

## References

1. White JG, Southgate E, Thomson JN, Brenner S: **The structure of the nervous system of the nematode *Caenorhabditis elegans*.** *Philos Trans R Soc Lond Ser B Biol Sci* 1986, **314**:1–340, https://doi.org/10.1098/rstb.1986.0056.
2. Chen BL, Hall DH, Chklovskii DB: **Wiring optimization can relate neuronal structure and function.** *Proc Natl Acad Sci USA* 2006, **103**:4723–4728, https://doi.org/10.1073/pnas.0506806103.
3. Varshney LR, Chen BL, Paniagua E, Hall DH, Chklovskii DB: **Structural properties of the *Caenorhabditis elegans* neuronal network.** *PLoS Comput Biol* 2011, **7**, e1001066, https://doi.org/10.1371/journal.pcbi.1001066.
4. Brittin CA, Cook SJ, Hall DH, Emmons SW, Cohen N: **Volumetric reconstruction of main *Caenorhabditis elegans* neuropil at two different time points.** bioRxiv. 2018. https://doi.org/10.1101/485771. and data on, http://wormwiring.org.
   The complete volumetric reconstruction of *C. elegans* nerve ring reveals striking similarities with other nervous systems. Most neuron classes innervate well defined neighborhoods and aggregate functionally similar synapses to support distinct computational pathways. Rich-club neurons often change neighborhoods.
5. Ward S: **Chemotaxis by the nematode *Caenorhabditis elegans*: identification of attractants and analysis of the response by use of mutants.** *Proc Natl Acad Sci USA* 1973, **70**:817–821, https://doi.org/10.1073/pnas.70.3.817.
6. Iino Y, Yoshida K: **Parallel use of two behavioral mechanisms for chemotaxis in *Caenorhabditis elegans*.** *J Neurosci* 2009, **29**:5370–5380, https://doi.org/10.1523/JNEUROSCI.3633-08.2009.
7. Kocabas A, Shen C-H, Guo ZV, Ramanathan S: **Controlling interneuron activity in *Caenorhabditis elegans* to evoke chemotactic behaviour.** *Nature* 2012, **490**:273, https://doi.org/10.1038/nature11431.
8. Yeon J, Kim J, Kim D-Y, Kim H, Kim J, Du EJ, Kang KJ, Lim H-H, Moon D, Kim K: **A sensory-motor neuron type mediates proprioceptive coordination of steering in *C. elegans* via two TRPC channels.** *PLoS Biol* 2018, **16**:\[1]–\[26], https://doi.org/10.1371/journal.pbio.2004929.
   SMDD head motor neurons are identified as proprioceptive neurons, revealing an asymmetric proprioceptive circuit for steering. Proprioception in SMDD neurons is shown to be mediated by TRP-1 and TRP-2 cation channels.
9. Pierce-Shimomura JT, Morse TM, Lockery SR: **The fundamental role of pirouettes in *Caenorhabditis elegans* chemotaxis.** *J Neurosci* 1999, **19**:9557–9569, https://doi.org/10.1523/JNEUROSCI.19-21-09557.1999.
10. Gray JM, Hill JJ, Bargmann CI: **A circuit for navigation in *Caenorhabditis elegans*.** *Proc Natl Acad Sci USA* 2005, **102**:3184–3191, https://doi.org/10.1073/pnas.0409009101.
11. Broekmans OD, Rodgers JB, Ryu WS, Stephens GJ: **Resolving coiled shapes reveals new reorientation behaviors in *C. elegans*.** *eLife* 2016, **5**:e17227, https://doi.org/10.7554/eLife.17227.
12. Donnelly JL, Clark CM, Leifer AM, Pirri JK, Haburcak M, Francis MM, Samuel ADT, Alkema MJ: **Monoaminergic orchestration of motor programs in a complex *C. elegans* behavior.** *PLoS Biol* 2013, **11**, e1001529, https://doi.org/10.1371/journal.pbio.1001529.
13. Pierce-Shimomura JT, Dores M, Lockery SR: **Analysis of the effects of turning bias on chemotaxis in *C. elegans*.** *J Exp Biol* 2005, **208**:4727–4733, https://doi.org/10.1242/jeb.01933.
14. Berri S, Boyle JH, Tassieri M, Hope IA, Cohen N: **Forward locomotion of the nematode *C. elegans* is achieved through modulation of a single gait.** *HFSP J* 2009, **3**:186–193, https://doi.org/10.2976/1.3082260.
15. Fang-Yen C, Wyart M, Xie J, Kawai R, Kodger T, Chen S, Wen Q, Samuel ADT: **Biomechanical analysis of gait adaptation in the nematode *Caenorhabditis elegans*.** *Proc Natl Acad Sci USA* 2010, **107**:20323–20328, https://doi.org/10.1073/pnas.1003016107.
16. Lockery SR, Lawton KJ, Doll JC, Faumont S, Coulthard SM, Thiele TR, Chronis N, McCormick KE, Goodman MB, Pruitt BL: **Artificial dirt: microfluidic substrates for nematode neurobiology and behavior.** *J Neurophysiol* 2008, **99**:3136–3143, https://doi.org/10.1152/jn.91327.2007.
17. Park S, Hwang H, Nam S-W, Martinez F, Austin RH, Ryu WS: **Enhanced *Caenorhabditis elegans* locomotion in a structured microfluidic environment.** *PLoS One* 2008, **3**:e2550, https://doi.org/10.1371/journal.pone.0002550.
18. Kim JS, Kaiser M: **From *Caenorhabditis elegans* to the human connectome: a specific modular organization increases metabolic, functional and developmental efficiency.** *Phil Trans Roy Soc Lond B* 2014, **369**:20130529, https://doi.org/10.1098/rstb.2013.0529.
19. Milo R, Shen-Orr S, Itzkovitz S, Kashtan N, Chklovskii D, Alon U: **Network motifs: simple building blocks of complex networks.** *Science* 2002, **298**:824–827, https://doi.org/10.1126/science.298.5594.824.
20. Sporns O, Kötter R: **Motifs in brain networks.** *PLoS Biol* 2004, **2**:e369, https://doi.org/10.1371/journal.pbio.0020369.
21. Azulay A, Itskovits E, Zaslaver A: **The *C. elegans* connectome consists of homogenous circuits with defined functional**

## Page 9

roles. PLoS Comput Biol 2016, **12**, e1005021,

.

22. Meunier D, Lambiotte R, Bullmore ET: **Modular and hierarchically modular organization of brain networks**. *Front Neurosci* 2010, **4**:200, https://doi.org/10.3389/fnins.2010.00200.

23. Towlson EK, Vértes PE, Ahnert SE, Schafer WR, Bullmore ET: **The rich club of the C. elegans neuronal connectome**. *J Neurosci* 2013, **33**:6380–6387, https://doi.org/10.1523/JNEUROSCI.3784-12.2013.

24. van den Heuvel MP, Sporns O: **Rich-club organization of the human connectome**. *J Neurosci* 2011, **31**:15775–15786, https://doi.org/10.1523/JNEUROSCI.3539-11.2011.

25. Kato S, Kaplan HS, Schrödel T, Skora S, Lindsay T, Yemini E, Lockery S, Zimmer M: **Global brain dynamics embed the motor command sequence of Caenorhabditis elegans**. *Cell* 2015, **163**:656–669, https://doi.org/10.1016/j.cell.2015.09.034.
    Brain-wide calcium imaging of the head in freely behaving animals, captured by principle component analysis of the time derivative of the calcium traces, allows for a direct mapping between global neural state and motor behavior.

26. Towlson EK, Vértes PE, Yan G, Chew YL, Walker DS, Schafer WR, Barabási AL: **Caenorhabditis elegans and the network control framework–FAQs**. *Philos Trans R Soc Lond Ser B Biol Sci* 2018, **373**:20170372, https://doi.org/10.1098/rstb.2017.0372.

27. Chalfie M, Sulston JE, White JG, Southgate E, Thomson JN, Brenner S: **The neural circuit for touch sensitivity in Caenorhabditis elegans**. *J Neurosci* 1985, **5**:956–964, https://doi.org/10.1523/JNEUROSCI.05-04-00956.1985.

28. Mellem JE, Brockie PJ, Madsen DM, Maricq AV: **Action potentials contribute to neuronal signaling in C. elegans**. *Nat Neurosci* 2008, **11**:865, https://doi.org/10.1038/nn.2131.

29. Roberts WM, Augustine SB, Lawton KJ, Lindsay TH, Thiele TR, Izquierdo EJ, Faumont S, Lindsay RA, Britton MC, Pokala N, et al.: **A stochastic neuronal model predicts random search behaviors at multiple spatial scales in C. elegans**. *eLife* 2016, **5**:e12572, https://doi.org/10.7554/eLife.12572.

30. Gordus A, Pokala N, Levy S, Flavell SW, Bargmann CI: **Feedback from network states generates variability in a probabilistic olfactory circuit**. *Cell* 2015, **161**:215–227, https://doi.org/10.1016/j.cell.2015.02.018.

31. Grabow C, Grosskinsky S, Timme M: **Speed of complex network synchronization**. *Eur Phys J B* 2011, **84**:613–626, https://doi.org/10.1140/epjb/e2011-20038-9. https://doi.org/10.1140/epjb/e2011-20038-9.

32. Yan G, Vértes PE, Towlson EK, Chew YL, Walker DS, Schafer WR, Barabási AL: **Network control principles predict neuron function in the Caenorhabditis elegans connectome**. *Nature* 2017, **550**:519, https://doi.org/10.1038/nature24056.

33. Zamora-López G, Zhou C, Kurths J: **Exploring brain function from anatomical connectivity**. *Front Neurosci* 2011, **5**:83, https://doi.org/10.3389/fnins.2011.00083.

34. Cabral J, Kringelbach ML, Deco G: **Exploring the network dynamics underlying brain activity during rest**. *Prog Neurobiol* 2014, **114**:102–131. https://doi.org/10.1016/j.pneurobio.2013.12.005.

35. Nichols ALA, Eichler T, Latham R, Zimmer M: **A global brain state underlies C. elegans sleep behavior**. *Science* 2017, **356**, https://doi.org/10.1126/science.aam6851. eaam6851.
    Calcium imaging of C. elegans head neurons during sleep and wakefulness is used to show that a majority of neurons are active during wakefulness but quiescent during sleep; a significant proportion of active neurons during sleep are GABAergic.

36. Schrödel T, Prevedel R, Aumayr K, Zimmer M, Vaziri A: **Brain-wide 3D imaging of neuronal activity in Caenorhabditis elegans with sculpted light**. *Nat Methods* 2013, **10**:1013–1020, https://doi.org/10.1038/nmeth.2637.

37. Nguyen JP, Shipley FB, Linder AN, Plummer GS, Liu M, Setru SU, Shaevitz JW, Leifer AM: **Whole-brain calcium imaging with cellular resolution in freely behaving Caenorhabditis elegans**. *Proc Natl Acad Sci USA* 2016, **113**:E1074–E1081, https://doi.org/10.1073/pnas.1507110112.

38. Kaplan HS, Nichols ALA, Zimmer M: **Sensorimotor integration in Caenorhabditis elegans: a reappraisal towards dynamic and distributed computations**. *Philos Trans R Soc Lond Ser B Biol Sci* 1758, **373**, https://doi.org/10.1098/rstb.2017.0371.

39. Kunert JM, Shlizerman E, Kutz JN: **Low-dimensional functionality of complex network dynamics: Neurosensory integration in the Caenorhabditis elegans connectome**. *Phys Rev E* 2014, **89**:052805, https://doi.org/10.1103/PhysRevE.89.052805.

40. Kunert JM, Proctor JL, Brunton SL, Kutz J: **Spatiotemporal feedback and network structure drive and encode Caenorhabditis elegans locomotion**. *PLoS Comput Biol* 2017, **13**, e1005303, https://doi.org/10.1371/journal.pcbi.1005303.
    Dimensional reduction of neural dynamics from simulations of a full-connectome, disembodies model produce a low dimensional representation that enables system-wide comparison of the neural dynamics pre- and post-ablation of locomotion motorneurons and interneurons.

41. Stephens GJ, Johnson-Kerner B, Bialek W, Ryu WS: **Dimensionality and dynamics in the behavior of C. elegans**. *PLoS Comput Biol* 2008, **4**, e1000028, https://doi.org/10.1371/journal.pcbi.1000028.

42. Yemini E, Jucikas T, Grundy LJ, Brown AEX, Schafer WR: **A database of Caenorhabditis elegans behavioral phenotypes**. *Nat Methods* 2013, **10**:877–879, https://doi.org/10.1038/nmeth.2560.

43. Brown AEX, Yemini EI, Grundy LJ, Jucikas T, Schafer WR: **A dictionary of behavioral motifs reveals clusters of genes affecting Caenorhabditis elegans locomotion**. *Proc Natl Acad Sci* 2013, **110**:791–796, https://doi.org/10.1073/pnas.1211447110.

44. Li K, Javer A, Keaveny EE, Brown AEX: **Recurrent neural networks with interpretable cells predict and classify worm behaviour**. *bioRxiv* 2017:222208, https://doi.org/10.1101/222208.

45. Bargmann CI, Marder E: **From the connectome to brain function**. *Nat Methods* 2013, **10**:483, https://doi.org/10.1038/nmeth.2451.

46. Tsalik EL, Hobert OH: **Functional mapping of neurons that control locomotory behavior in Caenorhabditis elegans: mathematical modeling and molecular genetics**. *J Neurobiol* 2003, **56**:178–197, https://doi.org/10.1002/neu.10245.

47. White JG, Southgate E, Thomson JN, Brenner S: **The structure of the ventral nerve cord of Caenorhabditis elegans**. *Philos Trans R Soc Lond Ser B Biol Sci* 1976, **275**:327–348, https://doi.org/10.1098/rstb.1976.0086.

48. Erdös P, Niebur E: **The neural basis of the locomotion of nematodes**. In *Statistical mechanics of neural networks*. Springer; 1990:253–267, https://doi.org/10.1007/3540532676_54.

49. Bryden JA, Cohen N: **A simulation model of the locomotion controllers for the nematode Caenorhabditis elegans**. In *From animals to animats 8: Proceedings of the Eighth International Conference on the Simulation of Adaptive Behavior*. MIT Press; 2004:183–192.

50. Bryden JA, Cohen N: **Neural control of Caenorhabditis elegans forward locomotion: the role of sensory feedback**. *Biol Cybern* 2008, **98**:339–351, https://doi.org/10.1007/s00422-008-0212-6.

51. Boyle JH, Berri S, Cohen N: **Gait modulation in C. elegans: an integrated neuromechanical model**. *Front Comput Neurosci* 2012, **6**:10, https://doi.org/10.3389/fncom.2012.00010.

52. Hall DH, The role of gap junctions in the C. elegans connectome, *Neurosci Lett*: https://doi.org/10.1016/j.neulet.2017.09.002.

53. Rakowski F, Srinivasan J, Sternberg PW, Karbowski J: **Synaptic polarity of the interneuron circuit controlling C. elegans locomotion**. *Front Comput Neurosci* 2013, **7**:128, https://doi.org/10.3389/fncom.2013.00128.

54. Rakowski F, Karbowski J: **Optimal synaptic signaling connectome for locomotory behavior in Caenorhabditis elegans: Design minimizing energy cost**. *PLoS Comput Biol* 2017, **13**, e1005834, https://doi.org/10.1371/journal.pcbi.1005834.

## Page 10

55. Piggott BJ, Liu J, Feng Z, Wescott SA, Xu XZS: **The neural circuits and synaptic mechanisms underlying motor initiation in C. elegans**. *Cell* 2011, **147**:922–933, https://doi.org/10.1016/j.cell.2011.08.053.

56. Pereira L, Kratsios P, Serrano-Saiz E, Sheftel H, Mayo AE, Hall DH, White JG, LeBoeuf B, Garcia LR, Alon U, Hobert OH: **A cellular and regulatory map of the cholinergic nervous system of C. elegans**. *eLife* 2015, **4**:e12432, https://doi.org/10.7554/eLife.12432.

57. McIntire SL, Jorgensen E, Kaplan J, Horvitz HR: **The GABAergic nervous system of Caenorhabditis elegans**. *Nature* 1993, **364**: 337–341, https://doi.org/10.1038/364337a0.

58. Haspel G, O’Donovan MJ, Hart AC: **Motoneurons dedicated to either forward or backward locomotion in the nematode Caenorhabditis elegans**. *J Neurosci* 2010, **30**:11151–11156, https://doi.org/10.1523/JNEUROSCI.2244-10.2010.

59. Collins KM, Bode A, Fernandez RW, Tanis JE, Brewer JC, Creamer MS, Koelle MR: **Activity of the C. elegans egg-laying behavior circuit is controlled by competing activation and feedback inhibition**. *eLife* 2016, **5**:e21126, https://doi.org/10.7554/eLife.21126.

60. Tolstenkov O, Van der Auwera P, Costa WS, Bazhanova O, Gemeinhardt TM, Bergs AC, Gottschalk A: **Functionally asymmetric motor neurons contribute to coordinating locomotion of Caenorhabditis elegans**. *eLife* 2018, **7**:e29913, https://doi.org/10.7554/eLife.34997.
    * AS motoneurons are found to have roles in regulating dorsoventral bends and in gating forward and backward locomotion in C. elegans.

61. Fouad AD, Teng S, Mark JR, Liu A, Alvarez-Illera P, Ji H, Du A, Bhirgoo PD, Cornblath E, Guan SA, et al.: **Distributed rhythm generators underlie Caenorhabditis elegans forward locomotion**. *eLife* 2018, **7**:e29913, https://doi.org/10.7554/eLife.29913.
    Neuronal ablation experiments, optogenetics and severing of the ventral and dorsal nerve cords are used to show that isolating sections of the motor circuit produces oscillations at different frequencies along the body. Ablating most command inteneurons does not eliminate body oscillations, which require B-class motoneurons.

62. Boyle JH, Berri S, Tassieri M, Hope IA, Cohen N: **Gait modulation in C. elegans: it’s not a choice, it’s a reflex!**, Front Behav Neurosci 5. https://doi.org/10.3389/fnbeh.2011.00010.

63. Denham JE, Ranner T, Cohen N: **Signatures of proprioceptive control in Caenorhabditis elegans locomotion**. *Philos Trans R Soc Lond Ser B Biol Sci* 2018, **373**:20180208, https://doi.org/10.1098/rstb.2018.0208.
    An integrated neuromechanical model of C. elegans is used to draw qualitative distinctions between gait adaptation resulting from mechanical modulation and neural modulation. Kinematic signatures of proprioceptive versus central pattern generated control are identified.

64. Xu T, Huo J, Shao S, Po M, Kawano T, Lu Y, Wu M, Zhen M, Wen Q: **Descending pathway facilitates undulatory wave propagation in Caenorhabditis elegans through gap junctions**. *Proc Natl Acad Sci USA* 2018, **115**:E4493–E4502, https://doi.org/10.1073/pnas.1717022115. https://doi.org/10.1073/pnas.1717022115.

65. Cohen N, Sanders T: **Nematode locomotion: dissecting the neuronal–environmental loop**. *Curr Opin Neurobiol* 2014, **25**: 99–106, https://doi.org/10.1016/j.conb.2013.12.003.

66. Cohen N, Boyle JH: **Swimming at low Reynolds number: a beginners guide to undulatory locomotion**. *Contemp Phys* 2010, **51**:103–123, https://doi.org/10.1080/00107510903268381.

67. Haspel G, O’Donovan MJ: **A perimotor framework reveals functional segmentation in the motoneuronal network controlling locomotion in Caenorhabditis elegans**. *J Neurosci* 2011, **31**:14611–14623, https://doi.org/10.1523/JNEUROSCI.2186-11.2011.

68. Haspel G, O’Donovan MJ: **A connectivity model for the locomotor network of Caenorhabditis elegans**. *Worm* 2012, **1**: 125–128, https://doi.org/10.4161/worm.19392.

69. Gjorgjieva J, Biron D, Haspel G: **Neurobiology of Caenorhabditis elegans locomotion: where do we stand?** *Bioscience* 2014, **64**:476–486, https://doi.org/10.1093/biosci/biu058.

70. Wen Q, Po MD, Hulme E, Chen S, Liu X, Kwok SW, Gershow M, Leifer AM, Butler V, Fang-Yen C: **Proprioceptive coupling within motor neurons drives C. elegans forward locomotion**. *Neuron* 2012, **76**:750–761, https://doi.org/10.1016/j.neuron.2012.08.039.

71. Olivares EO, Izquierdo EJ, Beer RD: **Potential role of a ventral nerve cord central pattern generator in forward and backward locomotion in Caenorhabditis elegans**. *Netw Neurosci* 2018, **2**:323–343, https://doi.org/10.1162/netn_a_00036.
    Neuromechanical model demonstrates that proprioceptively driven SMD and RMD motoneurons in the head along with posterior proprioception in the body with an anatomically realistic range is sufficient to generate and propagate undulations.

72. Izquierdo EJ, Beer RD: **An integrated neuromechanical model of steering in C. elegans**. In Proceeding of the European Conference on Artificial Life; 2015:199–206, https://doi.org/10.7551/978-0-262-33027-5-ch040.

73. Izquierdo EJ, Beer RD: **From head to tail: a neuromechanical model of forward locomotion in Caenorhabditis elegans**. *Philos Trans R Soc Lond Ser B Biol Sci* 2018, **373**:1–12, https://doi.org/10.1098/rstb.2017.0374.

74. Fieseler C, Kunert-Graf J, Kutz JN: **The control structure of the nematode Caenorhabditis elegans: neuro-sensory integration and propioceptive feedback**. *J Biomech* 2018, **74**:1–8, https://doi.org/10.1016/j.jbiomech.2018.03.046.

75. Niebur E, Erdös P: **Theory of the locomotion of nematodes**. *Biophys J* 1991, **60**:1132–1146, https://doi.org/10.1016/s0006-3495(91)82149-x. https://doi.org/10.1016/S0006-3495(91)82149-X.

76. Karbowski J, Schindelman G, Cronin CJ, Seah A, Sternberg PW: **Systems level circuit model of C. elegans undulatory locomotion: mathematical modeling and molecular genetics**. *J Comput Neurosci* 2008, **24**:253–276, https://doi.org/10.1007/s10827-007-0054-6.

77. Liu P, Chen B, Wang Z-W: **SLO-2 potassium channel is an important regulator of neurotransmitter release in Caenorhabditis elegans**. *Nat Commun* 2014, **5**:5155, https://doi.org/10.1038/ncomms6155.

78. Kawano T, Po MD, Gao S, Leung G, Ryu WS, Zhen M: **An imbalancing act: gap junctions reduce the backward motor circuit activity to bias C. elegans for forward locomotion**. *Neuron* 2011, **72**:572–586, https://doi.org/10.1016/j.neuron.2011.09.005.

79. Liu P, Chen B, Mailler R, Wang Z-W: **Antidromic-rectifying gap junctions amplify chemical transmission at functionally mixed electrical-chemical synapses**. *Nat Commun* 2017, **8**: 14818, https://doi.org/10.1038/ncomms14818.

80. Gao S, Guan SA, Fouad AD, Meng J, Kawano T, Huang Y-C, Li Y, Alcaire S, Hung W, Lu Y, Qi YB, Jin Y, Alkema M, Fang-Yen C, Zhen M: **Excitatory motor neurons are local oscillators for backward locomotion**. *eLife* 2018, **7**:e29915, https://doi.org/10.7554/eLife.29915.
    Neuronal ablation, electrophysiology and calcium imaging are used to propose pattern generation and coordination mechanisms in the backward locomotion circuit. Slow pacemaking oscillations are demonstrated in A-class motoneurons, even in immobilized animals with severed descending control.

81. Pierce-Shimomura JT, Chen BL, Mun JJ, Ho R, Sarkis R, McIntire SL: **Genetic analysis of crawling and swimming locomotory patterns in C. elegans**. *Proc Natl Acad Sci USA* 2008, **105**:20982–20987, https://doi.org/10.1073/pnas.0810359105.

82. Sznitman J, Shen X, Purohit PK, Arratia PE: **The effects of fluid viscosity on the kinematics and material properties of C. elegans swimming at low Reynolds number**. *Exp Mech* 2010, **50**:1303–1311, https://doi.org/10.1007/s11340-010-9339-1.

83. Lebois F, Sauvage P, Py C, Cardoso O, Ladoux B, Hersen P, Di Meglio JM: **Locomotion control of Caenorhabditis elegans through confinement**. *Biophys J* 2012, **102**:2791–2798, https://doi.org/10.1016/j.bpj.2012.04.051.

84. Gilpin W, Uppaluri S, Brangwynne CP: **Worms under pressure: Bulk mechanical properties of C. elegans are independent of the**

## Page 11

cuticle. Biophys J 2015, **108**:1887–1898,

.

.

85. Sznitman J, Purohit PK, Krajacic P, Lamitina T, Arratia PE: **Material Properties of *Caenorhabditis elegans* Swimming at Low Reynolds Number**. *Biophys J* 2010, **98**:617–626, https://doi.org/10.1016/j.bpj.2009.11.010. https://doi.org/10.1016/j.bpj.2009.11.010.

86. Park S-J, Goodman MB, Pruitt BL: **Analysis of nematode mechanics by piezoresistive displacement clamp**. *Proc Natl Acad Sci* 2007, **104**:17376–17381, https://doi.org/10.1073/pnas.0702138104.

87. Backholm M, Ryu WS, Dalnoki-Veress K: **Viscoelastic properties of the nematode *Caenorhabditis elegans*, a self-similar, shear-thinning worm**. *Proc Natl Acad Sci* 2013, **110**:4528–4533, https://doi.org/10.1073/pnas.1219965110.

88. Backholm M, Ryu WS, Dalnoki-Veress K: **The nematode *C. elegans* as a complex viscoelastic fluid**. *Eur Phys J E* 2015, **38**, https://doi.org/10.1140/epje/i2015-15036-1.

89. Cohen N, Ranner T: **A new computational method for a model of *C. elegans* biomechanics: Insights into elasticity and locomotion performance**. 2017. arXiv e-prints 1702.04988.

90. Petzold BC, Park S-J, Ponce P, Roozeboom C, Powell C, Goodman MB, Pruitt BL: ***Caenorhabditis elegans* body mechanics are regulated by body wall muscle tone**. *Biophys J* 2011, **100**:1977–1985, https://doi.org/10.1016/j.bpj.2011.02.035.

91. Montenegro-Johnson TD, Gagnon DA, Arratia PE, Lauga E: **Flow analysis of the low Reynolds number swimmer *C. elegans***. *Phys Rev Fluids* 2016, **1**, https://doi.org/10.1103/physrevfluids.1.053202.

92. Rabets Y, Backholm M, Dalnoki-Veress K, Ryu WS: **Direct measurements of drag forces in *C.*\~elegans crawling locomotion**. *Biophys J* 2014, **107**:1980–1987, https://doi.org/10.1016/j.bpj.2014.09.006. https://doi.org/10.1016/j.bpj.2014.09.006.

93. Backholm M, Kasper AKS, Schulman RD, Ryu WS, Dalnoki-Veress K: **The effects of viscosity on the undulatory swimming dynamics of *C. elegans***. *Phys Fluids* 2015, **27**:091901, https://doi.org/10.1063/1.4931795. https://doi.org/10.1063/1.4931795.

94. Walker DS, Chew YL, Schafer WR: **Genetics of behavior in *C. elegans***. In *The oxford handbook of invertebrate neurobiology*; 2017, https://doi.org/10.1093/oxfordhb/9780190456757.013.5.

95. Bentley B, Branicky R, Barnes CL, Chew YL, Yemini E, Bullmore ET, Vértes PE, Schafer WR: **The multilayer connectome of *Caenorhabditis elegans***. *PLoS Comput Biol* 2016, **12**, e1005283, https://doi.org/10.1371/journal.pcbi.1005283.

96. Gleeson P, Lung D, Grosu R, Hasani R, Larson SD: **c302: a multiscale framework for modelling the nervous system of *Caenorhabditis elegans***. *Philos Trans R Soc Lond Ser B Biol Sci* 1758, **373**, https://doi.org/10.1098/rstb.2017.0379.

97. Cantarelli M, Marin B, Quintana A, Earnshaw M, Court R, Gleeson P, Dura-Bernal S, Silver R, Idili G: **Geppetto: a reusable modular open platform for exploring neuroscience data and models**. *Philos Trans R Soc Lond Ser B Biol Sci* 1758, **373**, https://doi.org/10.1098/rstb.2017.0380.

98. Sarma GP, Lee CW, Portegys T, Ghayoomie V, Jacobs T, Alicea B, Cantarelli M, Currie M, Gerkin RC, Gingell S, et al.: **Openworm: overview and recent advances in integrative biological simulation of *Caenorhabditis elegans***. *Philos Trans R Soc Lond Ser B Biol Sci* 2018, **373**:20170382, https://doi.org/10.1098/rstb.2017.0382.

99. Blau A, Callaly F, Cawley S, Coffey A, De Mauro A, Epelde G, Ferrara L, Krewer F, Liberale C, Machado P, et al.: **The Si elegans project – the challenges and prospects of emulating *Caenorhabditis elegans***. In *Biomimetic and bio-hybrid systems*. Edited by Duff A, Lepora N, Mura A, Prescott T, Verschure P, Cham: Springer International Publishing; 2014: 436–438, https://doi.org/10.1007/978-3-319-09435-9_54.

100. Mujika A, Leškovský P, Álvarez R, Otaduy MA, Epelde G: **Modeling behavioral experiment interaction and environmental stimuli for a synthetic *C. elegans***. *Front Neuroinf* 2017, **11**:71, https://doi.org/10.3389/fninf.2017.00071.
