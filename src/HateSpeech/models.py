from django.db import models

# Create your models here.
class Detect(models.Model):
    sentence = models.CharField(max_length=100)
    probability = models.IntegerField()
    