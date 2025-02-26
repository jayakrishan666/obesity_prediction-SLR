from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect  # Import redirect function

# Function to redirect '/' to '/predict/'
def home_redirect(request):
    return redirect('predict_obesity')  # Redirect to the prediction page

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_redirect),  # Redirect homepage to /predict/
    path('predict/', include('prediction.urls')),
]
