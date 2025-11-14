# Fact2Fiction

This repository houses resources and materials associated with the research paper titled "Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System," presented at AAAI 2026.

The code and additional resources are available through [Our Website](https://trustworthycomp.github.io/Fact2Fiction/).

## Citation

We encourage users of this repository to cite our work in their research. Please use the following citation when referencing our paper:

```
@inproceedings{fact2fiction2026,
  title={Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System},
  author={He, Haorui and Li, Yupeng and Zhu, Bin Benjamin and Wen, Dacheng and Cheng, Reynold and Lau, Francis C. M.},
  booktitle={Proc.~of AAAI},
  year={2026},
}
```

## Abstract

State-of-the-art fact-checking systems combat misinformation at scale by employing autonomous LLM-based agents to decompose complex claims into smaller sub-claims, verify each sub-claim individually, and aggregate the partial results to produce verdicts with justifications (explanatory rationales for the verdicts).
The security of these systems is crucial, as compromised fact-checkers, which tend to be easily underexplored, can amplify misinformation.

This work introduces **Fact2Fiction**, the first poisoning attack framework targeting such agentic fact-checking systems.
Fact2Fiction mirrors the decomposition strategy and exploits system-generated justifications to craft tailored malicious evidences that compromise sub-claim verification.
Extensive experiments demonstrate that Fact2Fiction achieves 8.9\%--21.2\% higher attack success rates than state-of-the-art attacks across various poisoning budgets. Fact2Fiction exposes security weaknesses in current fact-checking systems and highlights the need for defensive countermeasures.

