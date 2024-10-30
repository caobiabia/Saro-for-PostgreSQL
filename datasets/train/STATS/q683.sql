select  count(*) from postHistory as ph,  		posts as p,          users as u where ph.PostId = p.Id 	and p.OwnerUserId = u.Id  AND ph.CreationDate<='2014-09-09 18:00:05'::timestamp  AND p.Score<=62;
