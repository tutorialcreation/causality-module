from .views import PredictViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('predict', PredictViewSet, basename='predict')
urlpatterns = router.urls