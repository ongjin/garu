# Dictionary data notices

The Garu engine and package code are licensed under Apache-2.0. The imported dictionary entries embedded in Section 6 of `base.gmdl` retain the terms of their sources. The other model sections are unchanged by this dictionary import.

`bulk-provenance.json` identifies the imported words, source records, snapshots and model hash. The source list is maintained in `training/codebook_data/bulk_supplement.tsv` in the Garu repository. Our selection and normalization of this combined imported list is offered under CC BY-SA 2.0 KR; this does not replace the original source licenses or the rights granted for the MeCab portions under Apache-2.0.

| Source | Attribution and snapshot | Terms |
|---|---|---|
| [mecab-ko-dic](https://bitbucket.org/eunjeon/mecab-ko-dic) | Eunjeon project; Yongwoon Lee, Yungho Yu and credited contributors. Commit `f0e76b67ca2eeca45732c909b587d797f2cc70a7`. | [Apache-2.0](licenses/Apache-2.0.txt). Original [authors](licenses/MeCab-AUTHORS.txt) and [contributors](licenses/MeCab-CONTRIBUTORS.txt) are retained. |
| [한국어기초사전](https://krdict.korean.go.kr/) | National Institute of Korean Language (국립국어원), October 2026 full export. | [CC BY-SA 2.0 KR](licenses/CC-BY-SA-2.0-KR.txt); [source policy](https://krdict.korean.go.kr/kor/kboardPolicy/copyRightTermsInfo). |
| [우리말샘](https://opendict.korean.go.kr/) | National Institute of Korean Language and contributors credited by the source; reviewed-information export dated October 4, 2026. Per-entry source links are included in the provenance file. | [CC BY-SA 2.0 KR](licenses/CC-BY-SA-2.0-KR.txt); [source policy](https://opendict.korean.go.kr/service/copyrightPolicy). |

The changes comprise Unicode normalization, POS mapping, deduplication, selection based on morphology and corpus checks, assignment of a conservative dictionary weight, and FST compilation. The imported list contains word/POS data; it does not include dictionary definitions, example sentences, multimedia, or original corpus paragraphs. Source dictionary costs are not reused as Garu frequencies.

Retain the applicable attribution and modification notices when redistributing this data. Include a copy of the Apache-2.0 license for the MeCab-derived portions, and the CC license text or its URI for the CC-licensed portions. The ShareAlike condition applies to adaptations of the CC-licensed data. This notice does not relicense the independent engine code or grant new rights to the original training corpora. The full license texts govern their respective source material.

The context-POS parameters in Section 16, the optional `garu-ko/text` boundary weights and the Q4 context-verifier weights in `garu-ko/sentences` were trained from the National Institute of Korean Language (국립국어원) NIKL morphological corpus, 2021 edition. Since research history #83 the sentence half of the `garu-ko/text` boundary weights also uses Instagram sentence records from the NIKL online posting materials corpus 2025 (국립국어원 온라인 게시 자료 말뭉치 2025, version 1.0), and the Section 17 spoken `그` rule uses the 2021 spoken corpus with tildes removed. Training and evaluation lineage is recorded in `training/ambiguity/` and `training/boundaries/` in the repository; `sentence-provenance.json` identifies both the fast and morphology-assisted Q4 artifacts. Original corpus paragraphs and annotations are not included in the package.
