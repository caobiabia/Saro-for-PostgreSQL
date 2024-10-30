select  count(*) from comments as c,  		postHistory as ph where c.UserId = ph.UserId  AND c.Score=1  AND c.CreationDate>='2010-07-20 19:12:45'::timestamp;
