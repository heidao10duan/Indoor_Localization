from django.contrib import admin

from DeviceManagement.models import Device, DeviceGroup, Ads

# Register your models here.


@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ('mac', 'floor')


@admin.register(DeviceGroup)
class DeviceGroupAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')


@admin.register(Ads)
class AdsAdmin(admin.ModelAdmin):
    list_display = ('id', 'ads_url', 'ads_image', 'ads_context', 'mac', 'ads_duration')