select  count(*) from comments as c,  		badges as b where c.UserId = b.UserId  AND b.Date>='2010-07-19 19:39:09'::timestamp  AND b.Date<='2014-08-13 21:58:42'::timestamp  AND c.Score=0  AND c.CreationDate>='2010-08-01 15:34:01'::timestamp;