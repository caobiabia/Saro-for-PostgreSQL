select  count(*) from comments as c,  		posts as p,          postHistory as ph where p.Id = c.PostId 	and p.Id = ph.PostId  AND c.Score=0  AND c.CreationDate>='2010-07-19 20:46:12'::timestamp  AND ph.CreationDate>='2010-12-29 16:39:58'::timestamp  AND ph.CreationDate<='2014-08-07 22:06:24'::timestamp;