import numpy as np
import pandas as pd
import torch

from src.data_loader import DataBundle, TabularPreprocessor, TabularDataset
from src.modality import select_tcga_modality


def test_modality_controls_keep_unknown_genomics_and_identical_partitions():
    frame = pd.DataFrame({"case_submitter_id": ["a", "b", "c", "d"], "diagnoses.ajcc_pathologic_n": [0,1,0,1],
                          "age": [30.,40.,50.,60.], "stage": ["T1","T2","T1","T2"],
                          "genomic_mutation__BRAF": [1.,np.nan,np.nan,1.]})
    processor = TabularPreprocessor(numeric_columns=("age",), categorical_columns=("stage",),
                                   binary_columns=("genomic_mutation__BRAF",)).fit(frame)
    from torch.utils.data import DataLoader
    loader = DataLoader(TabularDataset(processor.transform(frame)), batch_size=4)
    bundle = DataBundle(loader, loader, loader, frame, frame, frame, frame, frame, processor, {})
    clinical = select_tcga_modality(bundle, "clinical", 4)
    genomic = select_tcga_modality(bundle, "genomic", 4)
    assert all("genomic" not in feature for feature in clinical.preprocessor.feature_names)
    assert genomic.preprocessor.feature_names == ["genomic_mutation__BRAF", "missing__genomic_mutation__BRAF"]
    np.testing.assert_array_equal(genomic.test_loader.dataset.features[:,1], [0,1,1,0])
    assert clinical.test_frame is genomic.test_frame is bundle.test_frame
    assert torch.equal(clinical.test_loader.dataset.target, genomic.test_loader.dataset.target)
