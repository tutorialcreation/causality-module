from django.contrib.auth.models import User
from .models import CancerMeasure
from django.shortcuts import get_object_or_404
from .serializers import PredictSerializer
from rest_framework import viewsets
from rest_framework.response import Response

# Create your views here.
class PredictViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing user instances.
    """
    serializer_class = PredictSerializer
    queryset = CancerMeasure.objects.all()