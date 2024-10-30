select  count(*) from comments as c,  		badges as b where c.UserId = b.UserId  AND c.Score=0;
