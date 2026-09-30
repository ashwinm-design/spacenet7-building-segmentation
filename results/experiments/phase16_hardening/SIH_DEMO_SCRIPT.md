# SIH26012 — 5-Minute Live Presentation Script

**Project Title:** High-Resolution Aerial & Drone Building Footprint Segmentation and Cadastral Feature Extraction  
**Problem Statement ID:** SIH26012  
**Presenter Guide & Timing:** Exactly 5 Minutes (0:00 – 5:00)  

---

### [0:00 – 0:30] Problem Statement & Real-World Context
> *"Good morning, esteemed judges. Over 70% of developing urban areas lack up-to-date, digitized cadastral records. Traditional ground surveys take months to map a single ward, leading to boundary disputes and untaxed properties. Satellite imagery alone lacks the resolution to distinguish parcel access corridors.  
> Our project, **SIH26012**, bridges this gap by creating an end-to-end, AI-assisted cadastral reconnaissance pipeline: from drone and aerial orthomosaics down to preliminary parcel candidates with a human-in-the-loop WebGIS verification platform."*

---

### [0:30 – 1:00] Data Ingestion & Input Validation
> *"Let us launch a live demonstration. We open our WebGIS interface and upload a high-resolution GeoTIFF orthomosaic—here from the Open Cities dataset.  
> As we click 'Run AI Analysis', the system validates raster integrity in milliseconds: checking bands, coordinate systems (UTM EPSG:32732), and absence of NaN artifacts. The backend processes the tile asynchronously while updating our real-time 10-stage checklist."*

---

### [1:00 – 1:30] AI Building Footprint Segmentation
> *"Stage 3 and 4 execute our 3-Class Convolutional Neural Network trained on SpaceNet 7 and adapted zero-shot. Rather than naive binary masks, our model predicts **Building Body**, **Building Boundary**, and **Background**.  
> This separation prevents touching buildings from merging into monolithic blobs. Within 3.5 seconds, our polygonizer vectors the contours, applies Douglas-Peucker simplification, and extracts 254 distinct building footprints with instance-level area, perimeter, and confidence metrics."*

---

### [1:30 – 2:00] Dual-Path Road Extraction & Hybrid Fusion
> *"Cadastral boundaries cannot be inferred without understanding access networks. To achieve true robustness, we do not rely on deep learning alone.  
> In Stage 5 and 6, our pipeline runs a **dual extraction engine**: classical morphological CV for physical corridors and a ResNet18 U-Net for learned road semantics.  
> In Phase 13D hybrid fusion, we combine both into a topological graph, recovering 4 continuous road corridors totaling 1,571 meters of unobstructed access."*

---

### [2:00 – 2:45] Preliminary Parcel Candidate Generation
> *"Now comes the core innovation: how do we infer parcels without legal cadastre ground truth?  
> Using our building centroids as spatial seeds and the fused road corridors as strict physical barriers, we execute road-constrained Voronoi tessellation.  
> Every parcel candidate is clipped against road buffer corridors and passed through Shapely geometry validation. The result: 254 preliminary parcel boundaries with **100.0% topological validity** and zero road overlap."*

---

### [2:45 – 3:30] Parcel Intelligence & Uncertainty Quantification
> *"Not all AI boundaries are equal. In Stage 8, our **Parcel Intelligence Engine** evaluates every parcel across 4 spatial dimensions:
> 1. Building footprint containment evidence
> 2. Direct road frontage and access distance
> 3. Polsby-Popper compactness and geometric regularity
> 4. Topological relationship with neighboring parcels  
> The engine assigns a composite confidence score (averaging 88.2%) and automatically flags 210 candidates with uncertainty flags like `WEAK_BUILDING_EVIDENCE` or `MULTI_SOURCE_CONFLICT` to guide surveyor attention."*

---

### [3:30 – 4:15] Human-in-the-Loop WebGIS Verification
> *"We click `[ OPEN WEBGIS ]` and enter the verification workflow.  
> Notice Candidate **#56**: it has a lower confidence of 62.5% and a flag indicating zero road frontage. As a municipal surveyor, we inspect the satellite view and mark it as **'Needs Review'** with a note for field inspection.  
> Next, we inspect Candidate **#159**: 100% confidence, pristine geometry, and 20 meters of road frontage. We click **'Accept'**.  
> If an operator needs to adjust a boundary, they click **'Edit'**, drag the vertices interactively, and save. Every action is recorded in an immutable, timestamped audit log."*

---

### [4:15 – 4:45] GIS-Ready Export & Interoperability
> *"Municipalities need data in standard formats. In our Export Center, we can download with one click:
> - Building footprints as GeoJSON
> - Fused road network corridors as GeoJSON
> - Preliminary and human-reviewed parcels as GeoJSON
> - Complete tabular metrics as CSV
> - Complete analysis summary and audit trail as JSON  
> These files load directly into QGIS, ArcGIS, or municipal land registries."*

---

### [4:45 – 5:00] Limitations & Real-World Impact
> *"Crucially, our system does **not** claim legal title ownership—AI generates spatial hypotheses that empower licensed surveyors, reducing field reconnaissance time by up to 10x.  
> By uniting modern computer vision, computational geometry, and verifiable human oversight into a modular, production-ready stack, SIH26012 makes AI-assisted urban parcel mapping practical, transparent, and reproducible today. Thank you!"*
