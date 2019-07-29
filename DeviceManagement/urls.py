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
from . import views_ads


urlpatterns = [
    url(r'^add_group/', views.add_group, name='add_group'),
    url(r'^get_group/', views.get_group, name='get_group'),
    url(r'^delete_group/', views.delete_group, name='delete_group'),
    url(r'^select_group/', views.select_group, name='select_group'),
    url(r'^add_device/', views.add_device, name='add_device'),
    url(r'^get_device/', views.get_device, name='get_device'),
    url(r'^select_device/', views.select_device, name='select_device'), # 设备下拉框

    url(r'^push_ads/', views_ads.push_ads, name='push_ad'), # 广告推送配置
    url(r'^add_ads/', views_ads.add_ads, name='add_ads'), # 添加广告
    url(r'^get_ads/', views_ads.get_ads, name='get_ads'),
    url(r'^delete_ads/', views_ads.delete_ads, name='delete_ads'),
]
