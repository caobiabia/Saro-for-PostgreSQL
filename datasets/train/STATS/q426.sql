select  count(*) from badges as b, 		posts as p where b.UserId = p.OwnerUserId  AND p.Score>=-2  AND p.Score<=12  AND p.AnswerCount>=0  AND p.AnswerCount<=4;
