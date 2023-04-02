---
title: 'Natural product biosynthesis 1. Part 1'
date: 2022-11-07
permalink: /posts/2012/08/natural-product-biosynthesis-pks/
tags:
  - biochemistry
  - mechanisms
  - chemical biology
  - protein
---
<style> body {text-align: justify} </style>

### Introduction 

There are upwards of 10^5 natural products that can be synthesised from simple building blocks [1]. These precursors are often compounds with a low diversity of atoms (mostly C, H and O) which can assemble into complex molecules through simple biosythetic routes. The complexity can arise through the simple repetition of these reactions. Although simple, these reactions are often challenging to synthetically perform because of their slow kinetics under standard laboratory conditions. Not only do biological enzymes funnel precursors down a shute of otherwise thermodynamically impossible reaction routes - such as hydride transfers and intramolecular cyclisations - they also exhibit exquisite stereoselectivity. Although simple, this perfectly executed catalysis is essential when the number of reaction steps becomes large - as is the case for more complex biological compounds. An understanding of these biological mechanisms is thus important from a syntehtic chemisry perspective and for exploiting biological pathways to generate novel natural products.

This post will describe polyketide synthases (PKSs). These are large megasynthases (>1.1 MDa), majorily found in fungi, that are arranged in  modular assembly lines. Each module ascribes a new building block to be joined onto the growing compound and is modular in that different enzymes can be inserted into the assembly line to further decorate the compound with greater functionality e.g hydroxyl, methoxymethyl and glycosylation groups. I hope that there is a general theme of the emergence of complexity through simple repetition. 


### General biological mechanisms

Mechanisms covered:
* Addition reactions e.g. nucleophillic and electrophillic 
* Condensation and hydrolysis reactions e.g. aldol and claisen condensations
* Elimination and isomerisation reactions e.g. enolisation and dehydrations


#### Enolisation

Isomerisation involves a rearangment that leaves the overall molecular weight of a compound unaltered. Enolisation (a type of tautomerisation involving proton transfers) is a type of isomerisation that occurs frequently in biochemistry and involves the interconversion between a ketone and an enol (Fig. 1). The two are in equilibrium and can be used by nature to drive reactions forward. These are commonly found in PKSs in order to form aromatic rings from cyclic chains. 

![Figure 1. Example of an enolisation mechanism ](/images/keto_enol_tautomerisation.png)*Figure 1. Example of an enolisation mechanism. A base abstracts a proton and a pair of electrons then gravitates towards the delta positive carbon of the carbonyl group to form a double bond. This repels electrons within the carbonyl group to accept a proton to stabilise the growing negative charge. *

#### Claisen condensation

A claisen condensation involves the condensation of esters. Decarboxylative driven condensations by decaboxylases are common. Rather than writing this mechanism in one go I often find it easier to visualise an overview of an mechanism by breaking down the mechanism into steps and tracking the movement of electrons. In biology, the ester reactants may be activated at an enzymes active site through formation of a thioester which is more reactive than an ester because sulfur is a stronger nucleophile. 

![Figure 2. Example of a claisen condensation between two esters](/images/claisen_condensation.png) *Figure 2. Example of a claisen condensation between two esters. (1) Loss of CO2 drives the reaction forward by forming an enolate ion. (2) This creates a double bond that acts as a nucleophile to attack the delta carbon of the carbonly group. (3) If there is a strong nucleophile bound to the carbonly this will be expelled via an enolate intermediate.*


#### Nucleophillic  acyl substitution reactions

Nucleophillic acyl subsitution reactions involve a nucleophile attacking a carbonyl group to generate a R-CONu (Fig.3). The reaction largely depends on the stability of the carbonyl group (Figure 2). Amides are the least reactive because resonance stabilisation makes NH2 a poor leaving group. Acyl phosphates are very reactive and divalent cations (Mg2+) can further stablilise the negatively charged phosphate (PO42-) to increase reactivity of the leaving group.

![Figure 3. Example of a nucleophillic acyl substitution mechanism](/images/nucleophillic_acyl_substitution.png) *Figure 3. Example of a nucleophillic acyl substitution mechanism. A nucleophile attacks the carbonly group with the subsequent movement of electrons within the carbonly group expelling the weaker nucleophillic leaving group (X).*

There can be two types of nucleophillic substitution reactions: (i) SN1 and (ii) SN2. These are named based on the rate limiting step of the reaction. In SN1 reactions the rate determining step is the loss of the leaving group (which in this case is often strong e.g. diphosphates). Once the leaving group is lost a, often weak, nucleophile can attack the carbocation. In SN2 reactions the rate is dependent on the concentration of the substrate and nucleophile. The nuclephile attacks furthest away from the leaving group (backside) in a single step reaction. This can result in inversion of the stereocentre if a chiral carbon is the recipient of the attack.



#### Elimination

Elimination reactions such as dehydrations are common in biochemistry and often proceed by an E1cb mechanism (Fig.4). This is favoured under basic conditions and involves poor leaving groups (C=0/O-R), although acidic hydrogens can increase their reactivity. Acidic hydrogens lie between strong electron withdrawing groups. The most acidic hydrogens would be those situated between two carboxylic acid groups e.g. those in a dicarboxylic acid, because they would have a strong tendency to be lost as H+ in order to stabilise the negatively charged groups.

![Figure 4. Example of an elimination mechanism](/images/elimination_reaction_example.png) *Figure 4. Example of an elimination mechanism. In this example there is a carbonly group which is a poor leaving group because of its double bond. Protonation via a nucleophillic addition reaction involving NADPH as the reductant (a hydride ion is the nucleophile) could oxidise the carbonly to an alcohol which is a more reactive leaving group. A base catalysed reaction could then proceed involving the loss of water and the subsequent regeneration of the base by bulk solvent. The base may be a basic residue of an enzymes active site e.g. histidine, arginine or lysine (They are basic residues because they are all positively charged at physiological pH). Histidine is especially common in biological mechanisms because it can readily interconvert between neutral and +1 charged which makes it a good catalyst. 


#### Expoxidation 

Epoxidations generate cyclic ethers (Fig.5). In comparison to aliphatic ethers, their strained geometry makes cyclic ethers unusually reactive as nucleophiles, with the capability to generate long polyethers via a chain reaction of epoxidations (Fig.5; Lower). 

![Figure 5. Example of an expodiation mechanism](/images/epoxidation_mechanism_example.png). *Figure 5. Example of an epoxidation reaction (Upper mechanism). Here an alkene reacts with an ester to form a cyclic ether via a carbocation intermediate. You can see how the reaction could proceed in the reverse direction with the polyether being the nucleophile rather than the ether group (or how multiple cyclic ethers could be joined together as shown in the lower mechanism). I don't know if this is a real mechanism but it appears plausible to me and from it you can see how more complex molecules can begin to form from the repetition of simple mechanisms. 

### A minimal PKS assembly line 

A minimal PKS module requires three essential enzymes arranged linearly; ketosynthase (KS) acyltransferase (AT) and ACP domains, with each module elongating the compounds chain further. The inputs into this assembly line are secondary metabolites from metabolic processes e.g. acetly-coA and malonyl-coA are derived from the acetate pathway in lipid metabolism. A starter unit is loaded onto the loading module (module 0) and the extender unit is loaded onto the next module (module 1). These two units are then joined together and subsequently released from the PKS. This repetitive process forms long B-ketothioester chains (Beta: because the keto group of the precursor is 2nd from the carboxyl group; thioester; RC=OS-ACP).

Before one of these starter units can be loaded, the assembly line must be primed. It is also important to note that the following is the minimum number of domains required for a PKS. A PKS may have additional domains of a variety of functions e.g. methyltransferases and ketoreductases. 

The below steps describe Fig.6:

1. A long PPANT arm is transferred from co-enzyme A onto the ACP domain. This functions in mediating the transfer of the growing precursor between the different domains via transacylation reactions. 
2. The starter and extender units are loaded onto the AT domains of different modules via transesterification reactions (i.e. an ester is formed at the AT domain; RCOO-AT). The extender unit is transferred from the AT domain to the ACP domain and the starter unit is transferred from the AT domain onto the ACP domain and then subsequently onto the KS domain, all via transacylation reactions. 
3. When the starter and extender units are loaded onto the KS and ACP domains respectively, a claisen condensation reaction catalysed by the KS domain joins the two together with the expulsion of CO2.
4. This loading and extension process is repeated depending on the number of modules (KS, AT, ACP) of the PKS. When the growing polyketide chain reaches the terminating module it is transesterified onto a thioesterase domain. Here the polyketide can be released from the PKS in a number of ways, most commonly by hydrolysis or cyclisation of the chain. 
5. Tailoring enzymes, not 'bound' to the PKS assembly line, can further functionalise the polyketide.

![Figure 6. A minimal PKS assembly line](/images/PKS_overview.png) *Figure 6. A minimal PKS assembly line. Only the transesterification of the extender unit is shown, not the starter unit. 


### PKS example 1

Largely using the few mechanisms mentioned above we can build plausible retrosynthetic pathways for given polyketides. This can form the basis for beginning to dissect unknown PKSs. Experimental evidence, such as labelled feeding experiments of precursor compounds followed by NMR, is one example of how starter and extender compounds may be identified, mechanisms ruled out or ascertained, and the components of each module of the PKS elucidated. 

Retrosynthesis of the following compound was a problem I was given (Fig.7). I don't know the name of the compound, but this could be synthesised via the acetate pathway which yields acetyl-coA. From acetyl-coA we can also synthesise other building blocks such as malonyl-coA, butyryl-coA and methylmalonyl-coA to name a few. 

1. By labelling the compound we can begin to decipher the starter and extender precursors. This compound can be synthesised from 1x acetyl-coA starter unit and 4x malonyl-coA extender units (Figure 7). 

    ![Figure 7. Annotating compound A with plausible precursors](/images/compound_A_annotated.png). 


2. Acetyl-coA is a product of the acetate pathway. Malonyl-coA can be produced by a biotin dependent acyl-coA carboxylase (Fig.8). Carboxylation proceeds in an aldol reaction with CO2 via an enolate ion intermediate. 

   ![Figure 8. Synthesis of malonyl-coA](/images/malonyl_coA_synthesis.png)
   
   To synthesise methylmalonylcoA, a SAM dependent methylase enzyme adds a methyl group to acetyl-coA to generate           
   propionyl-coA (Fig.9). This is followed by the same aforementioned carboxylation dependent reaction to generate
   methylmalonlycoA.

   ![Figure 9. Synthesis of methylmalonyl-coA](/images/methyl_malonyl_coA.png). 

3. Numbering of the compounds ring identifies eight Cs, suggesting the compound is formed from a tetraketide (There is a methyl group sticking out but this is a side chain of a precursor). Repeated claisen condensation reactions (catalysed by KSs) between the starter unit and the extender units yields this product (Fig.10; step 3).
4. Next, we have to release the tetraketide form the PKS via the thioesterase domain. Deprotonation of the acidic hydrogens bound to the alpha-carbon creates a carboanion. This can act as a nucleophile in a nucleophillic acyl substitution reaction to cyclise the chain and concomitantly releases the compound from the PKS (Fig.10; step 4). 
5. A secondary cyclisation can now occur by tailoring - non PKS bound - enzymes. This could proceed via an enolisation reaction that generates the aromatic ring of compound A (Fig.10; step 5).


   ![Figure 10. Synthesis of compound A via a plausible PKS](/images/PKS_example_mechanism.png). 


The above compound A could be synthesised by the shown method. There are also other routes for the synthesis of this compound. The methyl group that I added by forming methylmalonyl-coA could have also been added by a SAM dependent methyltransferase domain. This would be similar to Fig.9, although the nucleophile would be a double bond formed by enolisation. Based on the mechanisms above the components of this PKS would be a typical loading module (AT-ACP) followed by three KS-AT-ACP modules. 

This post is getting quite long, so I am going to make a separate post on an interesting example of a PKS that is mentioned in the following papers; [Chemistry of a Unique Polyketide-like Synthase](https://pubs.acs.org/doi/10.1021/jacs.7b13297), [Biosynthesis of saxtoxin analogs: the unexpected pathway](https://pubs.acs.org/doi/10.1021/ja00333a062) and [Biosynthetic route towards saxitoxin and shunt pathway](https://www.nature.com/articles/srep20340). 


### References 

[1] https://en.wikipedia.org/wiki/Natural_product#:~:text=Depending%20on%20the%20sources%2C%20the,ranges%20between%20300%2C000%20and%20400%2C000.
[2] The Organic Chemistry of Biological Pathways by John McMurry
