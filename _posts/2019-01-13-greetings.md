---
title: "Flâneur's View"
date: 2019-01-13 22:00 +0900
categories: 
  - MISC
tags:
  - Flaneur
  - chitchat
  - 잡담
---

아래의 문제들을 해결하기 위한 Hive 테이블을 구성하는 많은 수의 작은 파일들을 적은 수의 큰 파일들로 병합하는 2가지 방법

## 쿼리를 사용한 방법: `INSERT OVERWRITE`

- 간단하게 Hive 쿼리를 통해서 테이블을 구성하는 작은 크기의 많은 수의 파일들을 합쳐주는 방법
- 먼저 MapReduce 작업 수를 설정
```sql
set mapred.reduce.tasks=1
```
- 그리고 아래 쿼리를 실행해서 테이블 내용을 읽어서 다시 작은 수의 파일들로 병합
```sql
insert overwrite table <table_name> select * from <table_name> limit 999999999
```

- `limit` 다음에는 쿼리 실행 결과 출력되는 레코드 수보다 큰 값을 지정

## Hive Merge 설정을 통한 방법

Hive 쿼리 실행 결과 출력 파일 수에 대한 설정

### `hive.merge` 설정

