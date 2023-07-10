#!/usr/bin/env bash
# -*- coding:utf-8 -*-

EXP_NAME=$1

for record_type in entity relation relation-boundary event record
do

  echo -e "\n==============>" Mean String ${record_type} "<=============="
  python3 scripts/summary_result.py -output_path ${EXP_NAME} -mean -reduce run -record ${record_type} -string

  echo -e "\n==============>" Mean Offset ${record_type} "<=============="
  python3 scripts/summary_result.py -output_path ${EXP_NAME} -mean -reduce run -record ${record_type}

  echo -e "\n==============>" Std Offset ${record_type} "<=============="
  python3 scripts/summary_result.py -output_path ${EXP_NAME} -std -reduce run -record ${record_type}

  echo -e "\n==============>" Offset ${record_type} "<=============="
  python3 scripts/summary_result.py -output_path ${EXP_NAME} -record ${record_type}

done
