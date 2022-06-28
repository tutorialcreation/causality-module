from rest_framework import serializers
from .models import CancerMeasure

class PredictSerializer(serializers.ModelSerializer):

    class Meta:
        model = CancerMeasure
        fields = '__all__'
