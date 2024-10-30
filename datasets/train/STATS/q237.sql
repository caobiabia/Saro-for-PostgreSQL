select  count(*) from comments as c,  		badges as b where c.UserId = b.UserId  AND b.Date>='2010-07-27 12:58:44'::timestamp  AND c.Score=0;
