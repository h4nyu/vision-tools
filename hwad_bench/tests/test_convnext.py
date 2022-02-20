
import timm

def test_model() -> None:
    model_names = timm.list_models("convnext*")
    print(model_names)
    # m = timm.create_model('mobilenetv3_large_100', pretrained=True)
