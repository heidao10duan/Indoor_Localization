# Generated by Django 2.1.7 on 2019-07-03 03:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('RoomManagement', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Device',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mac', models.CharField(max_length=50, verbose_name='mac地址')),
                ('name', models.CharField(max_length=50, null=True, verbose_name='名称')),
                ('position', models.CharField(default='0 0', max_length=20, verbose_name='设备位置')),
                ('group', models.IntegerField(verbose_name='设备分组')),
                ('type', models.IntegerField(choices=[(0, 'gateway'), (1, 'tag')], null=True, verbose_name='设备类型')),
                ('floor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='RoomManagement.BuildingFloor', verbose_name='建筑物楼层')),
            ],
            options={
                'verbose_name': '设备',
                'verbose_name_plural': '设备管理',
            },
        ),
        migrations.CreateModel(
            name='DeviceGroup',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50, null=True, verbose_name='组名称')),
            ],
            options={
                'verbose_name': '设备分组',
                'verbose_name_plural': '设备分组管理',
            },
        ),
    ]
