---
title: "Overview of DARPA AI Research in 모두연"
date: 2019-01-14 22:00 +0900
categories: 
  - ML
  - Seminar
tags:
  - seminar
  - 모두연
  - DARPA
  - resarch_trend
published: false
---



## Darpa

- Learning with Less Lable(LwLL)를 목표로 연구가 진행중



## Episci 프로젝트

- LwLL를 위해 AMASE라는 프레임워크를 준비중
- Aerospace & Defense 에서 활용할 수 있는 모델을 만드는 중.
  - Costly data, severe consequence가 특징
- "Small data is the future of AI"
- "Uncertainty really matters"
- Tactical AI
- Surprise Based Active Learning



  ### Domain-specific system on chip(DSSoC)

  - 서로 다른 장점을 가진 칩틀을 구성하여 다양한 특정 도메인에 최적화된 연산을 최소의 cost로 실행하도록 하는 연구
  - Criteria: 최소 5개의 application의 동작을 5ns 시간 내에 스케쥴링 처리 + power, latency, energy 등을 효율적으로
  - 강화학습을 이용한 Intelligent Scheduler를 구현하는 것이 목표
  - 실행시간 내에 Scheduler를 구현: Online-decision을 구현
  - 실제 데이터가 없이 시뮬레이션 데이터를 바탕으로 구현
  - OOD detection이 가능한 모델을 구현



## SafeAI Lab

- Uncertainty quantification, Out-of-distribution detection에 대한 연구 
- 관련 코드들 재현+reusability 향상 등
- higher api를 활용하는 걸 추구
- 현재는 predictive uncertainty model에 특화된 라이브러리 모듈 구현을 목표로 하는 중
