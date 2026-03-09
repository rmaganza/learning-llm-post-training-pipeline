def test_compare_models_import() -> None:
    from post_training_pipeline.evaluation.regression import compare_models, print_comparison_report

    # Smoke test - compare_models needs real model paths, so we just test import
    assert callable(compare_models)
    assert callable(print_comparison_report)
