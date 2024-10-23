select  count(*) from postHistory as ph,  		posts as p,          users as u where ph.PostId = p.Id 	and p.OwnerUserId = u.Id  AND u.Views>=0  AND u.CreationDate<='2014-09-06 03:46:42'::timestamp;
