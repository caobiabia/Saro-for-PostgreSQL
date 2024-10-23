select  count(*) from comments as c,  		badges as b where c.UserId = b.UserId  AND b.Date>='2010-07-19 19:39:08'::timestamp  AND c.CreationDate<='2014-08-28 16:38:23'::timestamp;
