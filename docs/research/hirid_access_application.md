# HiRID access application (PhysioNet project request)

## Project title

Combining expert models for ICU mortality prediction across hospitals: adaptive expert selection versus architecture synthesis

## Scientific abstract

Machine learning models that predict outcomes in intensive care often lose accuracy when they are applied in hospitals other than the one they were trained in. Different model families (gradient-boosted trees, recurrent, convolutional, attention-based and state-space networks) encode different assumptions about the data and may transfer differently, but it is unclear how best to combine them. This bachelor's thesis compares two strategies for building on a pool of trained expert models: adaptive expert selection, where a learned router chooses or weights the experts for each patient, and neural architecture search that assembles new networks from building blocks taken from the experts. The research question is whether architectures synthesised from expert components generalise better to unseen hospitals than selecting among the original experts, and at what computational cost.

We will extract harmonised hourly vital signs, laboratory values and treatments (4 static and 48 dynamic variables) from MIMIC-IV, HiRID and eICU-CRD using the open-source ricu and YAIB pipelines. The primary task is ICU mortality predicted after the first 24 hours of the stay; acute kidney injury is a secondary task. Models are trained on pooled MIMIC-IV and HiRID data, eICU-CRD is held out as an external test set, and leave-one-database-out experiments are run over all databases. HiRID is needed because it adds a second, European training domain with a different patient population, charting practice and sampling resolution, which is what allows the router and the search to be tested under real distribution shift. We report AUROC, AUPRC, calibration and computational cost, with pre-specified statistical comparisons. Only aggregate results will be published.

## Lay summary

Computer models can help intensive care teams spot patients who are at high risk of dying, but a model trained on data from one hospital often works less well in another hospital. In this student research project we test two ways of making such models more reliable. The first lets a computer decide, for each patient, which of several existing models to trust. The second automatically builds a new model by combining useful parts of the existing ones. We will train the models on anonymised records from a US hospital (MIMIC-IV) and from Bern University Hospital (HiRID), and test them on records from many other US hospitals (eICU) that the models have never seen. The Bern data are important because they show how much patients and record keeping differ between countries. The results will show which approach transfers better between hospitals and how much computing it costs. No patient will be identified, and the data will only be used for this research.
