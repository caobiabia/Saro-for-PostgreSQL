select  count(*) from comments as c,  		posts as p,          postHistory as ph where p.Id = c.PostId 	and p.Id = ph.PostId  AND p.Score<=81  AND p.CreationDate>='2010-07-23 08:18:52'::timestamp;
