select  count(*) from badges as b, 		posts as p where b.UserId = p.OwnerUserId  AND b.Date<='2014-09-12 19:44:59'::timestamp  AND p.Score>=-1  AND p.Score<=37  AND p.AnswerCount<=9;
