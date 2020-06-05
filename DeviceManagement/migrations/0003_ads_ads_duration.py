# Generated by Django 2.1.7 on 2019-07-26 02:32

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DeviceManagement', '0002_ads'),
    ]

    operations = [
        migrations.AddField(
            model_name='ads',
            name='ads_duration',
            field=models.DurationField(default=datetime.timedelta(0, 180), verbose_name='广告显示时长'),
        ),
    ]