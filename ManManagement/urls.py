"""MedicineStorageSystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf.urls import url

from . import views

app_name='room_management'

urlpatterns = [
    # 暂时先不保存定位结果
    url(r'^get_location/', views.get_location, name='get_location'),
    url(r'^get_navigation/', views.get_navigation, name='get_navigation'),
    url(r'^get_navigationtest/', views.get_navigationtest, name='get_navigationtest'),
    url(r'^get_destname/', views.get_destname, name='get_destname'), # 获取目的地名称
    # url(r'^add_building_type/', views.add_building_type, name='add_building_type'),
    url(r'^get_train_data/', views.get_train_data, name='get_train_data'),
]

