from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .model import probability
import json
# Create your views here.

def detectSpeech(request):
    if request.method=='GET':
        data=json.loads(request.body)
        isHateSpeech = probability(data['content'])
        return JsonResponse({"Probability": isHateSpeech})
    if request.method=='POST':
        return HttpResponse(status=400)