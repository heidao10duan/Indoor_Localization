from datetime import timedelta

from django.db import models

# Create your models here.、


# 设备分组，小基站分组，网关分组
# -----------------------------------------------------------------
class DeviceGroup(models.Model):
    name = models.CharField(max_length=50, null=True, verbose_name='组名称')

    class Meta:
        verbose_name = "设备分组"
        verbose_name_plural = "设备分组管理"

    def __str__(self):
        return self.name


# 设备信息：网关或标签
# ----------------------------------------------------------------------
class Device(models.Model):
    mac = models.CharField(max_length=50, verbose_name='mac地址')
    name = models.CharField(max_length=50, null=True, verbose_name='名称')
    floor = models.ForeignKey('RoomManagement.BuildingFloor', on_delete=models.CASCADE, verbose_name='建筑物楼层')
    position = models.CharField(max_length=20, default="0 0", verbose_name="设备位置")
    group = models.IntegerField(verbose_name='设备分组')
    GENDER_CHOICES = (
        (0, "gateway"),
        (1, "tag")
    )
    type = models.IntegerField(choices=GENDER_CHOICES, verbose_name='设备类型', null=True)

    class Meta:
        verbose_name = "设备"
        verbose_name_plural = "设备管理"

    def __str__(self):
        return self.name


# 广告
# ----------------------------------------------------------------------
class Ads(models.Model):
    ads_url = models.URLField(max_length=200,verbose_name="广告链接", null=True)
    ads_image = models.ImageField(verbose_name=u'广告图片', upload_to='AdsImage/%Y/%m/%d', null=True)
    ads_context = models.TextField(verbose_name="广告文本内容", null=True)
    ads_duration = models.IntegerField(verbose_name="广告显示时长", default='30')
    mac = models.CharField(max_length=50, verbose_name='mac地址')

    class Meta:
        verbose_name = "广告制作"
        verbose_name_plural = "广告制作管理"

    def __str__(self):
        return self.id







